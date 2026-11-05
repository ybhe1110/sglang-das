//! HTTP instrumentation uses the native frontend's label names and observes
//! active requests until their body finishes or the client disconnects.
use crate::metrics::{Event, Metrics};
use axum::{
    Router,
    body::Body,
    extract::{MatchedPath, Request, State},
    http::{StatusCode, header::CONTENT_TYPE},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};

use futures::StreamExt;

pub(super) fn mount(app: Router, metrics: Option<Metrics>) -> Router {
    let Some(metrics) = metrics else {
        return app;
    };
    let exporter = metrics.clone();
    let scrape = get(move || {
        let exporter = exporter.clone();
        async move {
            match exporter.scrape().await {
                Ok(bytes) => (
                    [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
                    bytes,
                )
                    .into_response(),
                Err(error) => {
                    tracing::error!(%error, "cannot collect Prometheus metrics");
                    StatusCode::SERVICE_UNAVAILABLE.into_response()
                }
            }
        }
    });
    app.route("/metrics", scrape.clone())
        .route("/metrics/", scrape)
        .layer(middleware::from_fn_with_state(metrics, track))
}

struct ActiveRequest {
    metrics: Metrics,
    path: String,
    method: String,
    key: Option<String>,
    status: Option<u16>,
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.metrics.record(Event::HttpEnd {
            path: std::mem::take(&mut self.path),
            method: std::mem::take(&mut self.method),
            key: self.key.take(),
            status: self.status,
        });
    }
}

async fn track(State(metrics): State<Metrics>, request: Request, next: Next) -> Response {
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map_or("__unmatched__", MatchedPath::as_str)
        .to_owned();
    let method = request.method().to_string();
    let key = request
        .headers()
        .get("x-smg-routing-key")
        .and_then(|h| h.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    metrics.record(Event::HttpStart {
        path: path.clone(),
        method: method.clone(),
        key: key.clone(),
    });
    let mut active = ActiveRequest {
        metrics,
        path,
        method,
        key,
        status: None,
    };
    let response = next.run(request).await;
    active.status = Some(response.status().as_u16());
    response.map(|body| {
        let mut stream = body.into_data_stream();
        Body::from_stream(async_stream::stream! {
            let _active = active;
            while let Some(chunk) = stream.next().await {
                yield chunk;
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn route_and_body_lifetime_follow_metrics_flag() {
        let request = || {
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap()
        };
        let off = mount(Router::new(), None);
        assert_eq!(
            off.oneshot(request()).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );

        let mut active = 0;
        let metrics = Metrics::start(None, move |events, _| {
            for e in events {
                match e {
                    Event::HttpStart { .. } => active += 1,
                    Event::HttpEnd { .. } => active -= 1,
                    _ => {}
                }
            }
            Ok(active.to_string().into_bytes())
        })
        .unwrap();
        let app = mount(Router::new(), Some(metrics.clone()));
        let response = app.oneshot(request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(metrics.scrape().await.unwrap(), b"1");
        drop(response);
        assert_eq!(metrics.scrape().await.unwrap(), b"0");
    }
}
