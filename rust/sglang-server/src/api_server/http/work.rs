//! Bounded HTTP preparation. Each lane owns its workers and accepts a fixed
//! number of bodies, including clients still uploading or consuming results.
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{StatusCode, header};
use bytes::BytesMut;
use core_affinity::CoreId;
use futures::StreamExt;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::message::config::ServerArgs;

const SHORT_BYTES: usize = 32 * 1024;
const LONG_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const RESPONSE_BYTES: usize = 16 * 1024 * 1024;
type Job = Box<dyn FnOnce() + Send>;

pub(crate) struct Lane {
    queue: flume::Sender<Job>,
    slots: Arc<Semaphore>,
    body_limit: usize,
    capacity: usize,
    rejected: std::sync::atomic::AtomicU64,
}

pub(crate) struct HttpWorkers {
    short: Lane,
    long: Lane,
    pub(crate) generation: Lane,
}

impl HttpWorkers {
    pub(crate) fn new(
        args: &ServerArgs,
        utility_cores: &[CoreId],
        generation_cores: &[CoreId],
        threads: &mut Vec<JoinHandle<()>>,
    ) -> std::io::Result<Self> {
        let mut utility = utility_cores.iter().copied().cycle();
        let mut generation = generation_cores.iter().copied().cycle();
        let workers = args.tokenizer_worker_num.max(1);
        Ok(Self {
            short: Lane::new(
                "utility-short",
                args.utility_workers,
                args.utility_capacity,
                SHORT_BYTES,
                &mut utility,
                threads,
            )?,
            long: Lane::new(
                "utility-long",
                args.utility_workers,
                args.utility_capacity,
                LONG_BYTES,
                &mut utility,
                threads,
            )?,
            generation: Lane::new(
                "chat-prepare",
                workers,
                workers * 8,
                64 * 1024 * 1024,
                &mut generation,
                threads,
            )?,
        })
    }

    pub(crate) fn utility(&self, req: &Request) -> &Lane {
        if content_length(req).is_some_and(|n| n <= SHORT_BYTES) {
            &self.short
        } else {
            &self.long
        }
    }

    pub(crate) fn metrics(&self, rank: u32) -> String {
        let mut metrics = String::new();
        for (name, lane) in [
            ("short", &self.short),
            ("long", &self.long),
            ("generation", &self.generation),
        ] {
            use std::fmt::Write;
            let used = lane.capacity - lane.slots.available_permits();
            let rejected = lane.rejected.load(std::sync::atomic::Ordering::Relaxed);
            let _ = writeln!(
                metrics,
                "sglang:rust_http_in_flight{{lane=\"{name}\",dp_rank=\"{rank}\"}} {used}"
            );
            let _ = writeln!(
                metrics,
                "sglang:rust_http_rejected_total{{lane=\"{name}\",dp_rank=\"{rank}\"}} {rejected}"
            );
        }
        metrics
    }
}

#[cfg(test)]
impl Default for HttpWorkers {
    fn default() -> Self {
        Self::new(&ServerArgs::default(), &[], &[], &mut Vec::new()).unwrap()
    }
}

fn content_length(req: &Request) -> Option<usize> {
    req.headers()
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

#[derive(Debug)]
pub(crate) struct WorkError(pub StatusCode, pub &'static str);

impl Lane {
    fn new(
        name: &str,
        workers: usize,
        capacity: usize,
        body_limit: usize,
        cores: &mut impl Iterator<Item = CoreId>,
        threads: &mut Vec<JoinHandle<()>>,
    ) -> std::io::Result<Self> {
        let (queue, rx) = flume::bounded::<Job>(capacity);
        for i in 0..workers {
            let rx = rx.clone();
            let core = cores.next();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("{name}-{i}"))
                    .spawn(move || {
                        if let Some(core) = core {
                            core_affinity::set_for_current(core);
                        }
                        while let Ok(job) = rx.recv() {
                            job();
                        }
                    })?,
            );
        }
        Ok(Self {
            queue,
            slots: Arc::new(Semaphore::new(capacity)),
            body_limit,
            capacity,
            rejected: Default::default(),
        })
    }

    /// Acquire before reading even a chunked upload. A dropped receiver marks a
    /// queued closure obsolete. Once started, the closure retains its lease.
    pub(crate) async fn run<T: Send + 'static>(
        &self,
        req: Request,
        work: impl FnOnce(Bytes, OwnedSemaphorePermit, &oneshot::Sender<T>) -> T + Send + 'static,
    ) -> Result<T, WorkError> {
        let mime = req
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if mime != "application/json"
            && !(mime.starts_with("application/") && mime.ends_with("+json"))
        {
            return Err(WorkError(
                StatusCode::BAD_REQUEST,
                "Expected request with JSON Content-Type",
            ));
        }
        let too_large = || {
            WorkError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request exceeds the HTTP preparation byte limit",
            )
        };
        if content_length(&req).is_some_and(|n| n > self.body_limit) {
            return Err(too_large());
        }
        let lease = self.slots.clone().try_acquire_owned().map_err(|_| {
            self.rejected
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            WorkError(
                StatusCode::SERVICE_UNAVAILABLE,
                "HTTP preparation capacity exhausted",
            )
        })?;
        tokio::time::timeout(Duration::from_secs(60), async {
            let mut stream = req.into_body().into_data_stream();
            let mut bytes = BytesMut::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk
                    .map_err(|_| WorkError(StatusCode::BAD_REQUEST, "incomplete request body"))?;
                if chunk.len() > self.body_limit - bytes.len() {
                    return Err(too_large());
                }
                bytes.extend_from_slice(&chunk);
            }
            let (tx, rx) = oneshot::channel();
            self.queue
                .try_send(Box::new(move || {
                    if !tx.is_closed() {
                        let result = work(bytes.freeze(), lease, &tx);
                        let _ = tx.send(result);
                    }
                }))
                .map_err(|_| {
                    WorkError(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "HTTP preparation workers unavailable",
                    )
                })?;
            rx.await
                .map_err(|_| WorkError(StatusCode::SERVICE_UNAVAILABLE, "HTTP preparation stopped"))
        })
        .await
        .map_err(|_| WorkError(StatusCode::GATEWAY_TIMEOUT, "HTTP preparation timed out"))?
    }
}

/// Hyper may hold the last data frame after dropping the Axum body. Attach the
/// permit to the actual bytes so even a slow socket is charged until release.
pub(crate) fn leased_body(data: Vec<u8>, lease: OwnedSemaphorePermit) -> Body {
    struct Reply {
        data: Vec<u8>,
        _lease: OwnedSemaphorePermit,
    }
    impl AsRef<[u8]> for Reply {
        fn as_ref(&self) -> &[u8] {
            &self.data
        }
    }
    Body::from(Bytes::from_owner(Reply {
        data,
        _lease: lease,
    }))
}

pub(crate) trait Codec: Send + Sync {
    fn encode(&self, text: &str, specials: bool) -> Result<Vec<i64>, String>;
    fn decode(&self, ids: &[u32], skip_specials: bool) -> Result<String, String>;
}

pub(crate) struct DynamoCodec {
    with_specials: dynamo_tokenizers::Tokenizer,
    without_specials: dynamo_tokenizers::Tokenizer,
}
impl DynamoCodec {
    pub(crate) fn new(
        with_specials: dynamo_tokenizers::Tokenizer,
        without_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            with_specials,
            without_specials,
        }
    }
}
impl Codec for DynamoCodec {
    fn encode(&self, text: &str, specials: bool) -> Result<Vec<i64>, String> {
        let tokenizer = if specials {
            &self.with_specials
        } else {
            &self.without_specials
        };
        tokenizer
            .encode(text)
            .map(|e| e.token_ids().iter().copied().map(i64::from).collect())
            .map_err(|e| e.to_string())
    }
    fn decode(&self, ids: &[u32], skip_specials: bool) -> Result<String, String> {
        self.with_specials
            .decode(ids, skip_specials)
            .map(String::from)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn request(size: usize) -> Request {
        Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, size)
            .body(Body::from(vec![b'x'; size]))
            .unwrap()
    }

    #[tokio::test]
    async fn cancelled_long_work_keeps_its_slot_without_delaying_short_or_generation() {
        let workers = Arc::new(
            HttpWorkers::new(
                &ServerArgs {
                    utility_capacity: 1,
                    ..Default::default()
                },
                &[],
                &[],
                &mut Vec::new(),
            )
            .unwrap(),
        );
        let (began, started) = oneshot::channel();
        let (release, barrier) = std::sync::mpsc::channel();
        let state = workers.clone();
        let task = tokio::spawn(async move {
            let req = request(SHORT_BYTES + 1);
            state
                .utility(&req)
                .run(req, move |_, _lease, _| {
                    began.send(()).unwrap();
                    barrier.recv().unwrap()
                })
                .await
        });
        started.await.unwrap();
        task.abort();
        let _ = task.await;
        let req = request(SHORT_BYTES + 1);
        assert!(matches!(
            workers.utility(&req).run(req, |_, _, _| ()).await,
            Err(WorkError(StatusCode::SERVICE_UNAVAILABLE, _))
        ));
        let req = request(1);
        assert!(workers.utility(&req).run(req, |_, _, _| ()).await.is_ok());
        assert!(
            workers
                .generation
                .run(request(1), |_, _, _| ())
                .await
                .is_ok()
        );
        release.send(()).unwrap();
        drop(
            tokio::time::timeout(Duration::from_secs(2), workers.long.slots.acquire())
                .await
                .unwrap()
                .unwrap(),
        );
    }

    #[tokio::test]
    async fn response_quota_follows_final_bytes_and_oversize_upload_is_rejected() {
        let workers = HttpWorkers::new(
            &ServerArgs {
                utility_capacity: 1,
                ..Default::default()
            },
            &[],
            &[],
            &mut Vec::new(),
        )
        .unwrap();
        let body = workers
            .short
            .run(request(1), |_, lease, _| leased_body(vec![0; 32], lease))
            .await
            .unwrap();
        let bytes = to_bytes(body, 32).await.unwrap();
        assert_eq!(workers.short.slots.available_permits(), 0);
        drop(bytes);
        assert_eq!(workers.short.slots.available_permits(), 1);
        let req = Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, 1)
            .body(Body::from(vec![0; SHORT_BYTES + 1]))
            .unwrap();
        assert!(matches!(
            workers.utility(&req).run(req, |_, _, _| ()).await,
            Err(WorkError(StatusCode::PAYLOAD_TOO_LARGE, _))
        ));
        assert_eq!(workers.short.slots.available_permits(), 1);
    }
}
