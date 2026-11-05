//! Export through the same Python multiprocess registry as the scheduler.
//! Only this bounded worker crosses the GIL; serving threads timestamp events.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::message::request::MetricOptions;
use crate::message::response::ResponseItem;

type Labels = BTreeMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TerminalUsage {
    pub cached: u32,
    pub details: Option<serde_json::Value>,
    pub spec_verify: u32,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub(crate) enum Event {
    HttpStart {
        path: String,
        method: String,
        key: Option<String>,
    },
    HttpEnd {
        path: String,
        method: String,
        key: Option<String>,
        status: Option<u16>,
    },
    First {
        labels: Labels,
        stream: bool,
        seconds: f64,
    },
    Interval {
        labels: Labels,
        seconds: f64,
        tokens: u64,
    },
    Finished {
        labels: Labels,
        stream: bool,
        seconds: f64,
        prompt: u32,
        tokens: u64,
        grammar: bool,
        usage: Option<TerminalUsage>,
    },
    Aborted,
}

type Scrape = Result<Vec<u8>, String>;
#[derive(Debug)]
enum Work {
    Record(Event),
    Scrape(oneshot::Sender<Scrape>),
}

#[derive(Clone, Debug)]
pub(crate) struct Metrics {
    tx: flume::Sender<Work>,
    header: Option<String>,
}

impl Metrics {
    pub fn start(
        header: Option<String>,
        mut apply: impl FnMut(&[Event], bool) -> Scrape + Send + 'static,
    ) -> std::io::Result<Self> {
        let (tx, rx) = flume::bounded(8192);
        std::thread::Builder::new()
            .name("prometheus".into())
            .spawn(move || {
                let mut events = Vec::with_capacity(256);
                let mut record_error = None;
                while let Ok(first) = rx.recv() {
                    let mut reply = None;
                    for cmd in std::iter::once(first).chain(rx.try_iter()).take(256) {
                        match cmd {
                            Work::Record(event) => events.push(event),
                            Work::Scrape(sender) => {
                                reply = Some(sender);
                                break;
                            }
                        }
                    }
                    // A scrape is a barrier: every earlier observation is visible.
                    if !events.is_empty()
                        && let Err(error) = apply(&events, false)
                    {
                        tracing::error!(%error, "could not record Prometheus observations");
                        record_error = Some(error);
                    }
                    if let Some(sender) = reply {
                        // A partially applied batch cannot be retried without
                        // double-counting. Report the missed update to the scraper.
                        let result = match record_error.take() {
                            Some(error) => Err(error),
                            None => apply(&[], true),
                        };
                        let _ = sender.send(result);
                    }
                    events.clear();
                }
            })?;
        Ok(Self { tx, header })
    }

    pub fn openai_options(
        &self,
        headers: &axum::http::HeaderMap,
        priority: Option<i64>,
    ) -> MetricOptions {
        let custom_labels = self
            .header
            .as_ref()
            .and_then(|name| headers.get(name))
            .and_then(|value| serde_json::from_slice(value.as_bytes()).ok());
        MetricOptions {
            custom_labels,
            priority,
            log_metrics: None,
        }
    }

    pub fn record(&self, event: Event) {
        let _ = self.tx.send(Work::Record(event));
    }

    pub async fn scrape(&self) -> Scrape {
        let (reply, result) = oneshot::channel();
        self.tx
            .send_async(Work::Scrape(reply))
            .await
            .map_err(|_| "exporter closed")?;
        result.await.map_err(|_| "exporter closed")?
    }
}

#[derive(Debug)]
pub(crate) struct Generation {
    metrics: Metrics,
    labels: Labels,
    start: Instant,
    last: Option<Instant>,
    total_tokens: u64,
    stream: bool,
    prefill: bool,
    grammar: bool,
}

impl Generation {
    pub fn new(
        metrics: Metrics,
        stream: bool,
        prefill: bool,
        grammar: bool,
        options: &MetricOptions,
    ) -> Arc<Mutex<Self>> {
        let mut labels = options.custom_labels.clone().unwrap_or_default();
        if let Some(priority) = options.priority {
            labels.insert("priority".into(), priority.to_string());
        }
        Arc::new(Mutex::new(Self {
            metrics,
            labels,
            start: Instant::now(),
            last: None,
            total_tokens: 0,
            stream,
            prefill,
            grammar,
        }))
    }

    pub fn observe(&mut self, item: &ResponseItem) {
        let (frame, done) = match item {
            ResponseItem::Frame(frame) => (frame, false),
            ResponseItem::Done(frame) => (frame, true),
            ResponseItem::Error(crate::utils::error::Error::Cancelled) => {
                self.metrics.record(Event::Aborted);
                return;
            }
            _ => return,
        };
        let now = Instant::now();
        let tokens = frame.completion_tokens;
        let elapsed = now.duration_since(self.start).as_secs_f64();
        if self.last.is_none() && !self.prefill {
            self.metrics.record(Event::First {
                labels: self.labels.clone(),
                stream: self.stream,
                seconds: elapsed,
            });
            self.last = Some(now);
        } else if tokens > 0 {
            // Like Python, prefill skips TTFT and starts its first interval at
            // the first output. Subsequent intervals are weighted by token count.
            let previous = self.last.unwrap_or(now);
            let observed = Instant::now();
            self.metrics.record(Event::Interval {
                labels: self.labels.clone(),
                seconds: observed.duration_since(previous).as_secs_f64(),
                tokens,
            });
            self.last = Some(observed);
        }
        self.total_tokens += tokens;
        if done {
            self.metrics.record(Event::Finished {
                stream: self.stream,
                seconds: elapsed,
                prompt: frame.prompt_tokens,
                labels: self.labels.clone(),
                tokens: self.total_tokens,
                grammar: self.grammar,
                usage: frame.usage.as_deref().cloned(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::response::ChunkEvent;

    #[tokio::test]
    async fn prefill_emits_terminal_metrics_without_decode_ttft() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        let metrics = Metrics::start(None, move |batch, _| {
            seen.lock()
                .unwrap()
                .extend(batch.iter().map(|e| serde_json::to_value(e).unwrap()));
            Ok(Vec::new())
        })
        .unwrap();
        let request = Generation::new(metrics.clone(), false, true, false, &Default::default());
        request
            .lock()
            .unwrap()
            .observe(&ResponseItem::Done(ChunkEvent {
                prompt_tokens: 12,
                completion_tokens: 1,
                usage: Some(Box::new(TerminalUsage {
                    cached: 8,
                    details: None,
                    spec_verify: 0,
                })),
                ..Default::default()
            }));
        metrics.scrape().await.unwrap();
        let samples = events.lock().unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0]["type"], "interval");
        assert_eq!(samples[1]["type"], "finished");
        assert_eq!(samples[1]["tokens"], 1);
        assert_eq!(samples[1]["usage"]["cached"], 8);
    }
}
