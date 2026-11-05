//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;

use axum::Router;

use super::disaggregation::bootstrap as pd_bootstrap;
use super::{common, loads, log, native_api, openai};
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::Senders;

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning each
/// `flume::Sender` and the chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) struct AppState {
    pub(super) senders: Senders,
    pub(super) response_buf: usize,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
    /// Response heartbeat (bumped per drained ring frame).
    pub(super) response_activity: ActivityCounter,
}

pub(crate) enum BootstrapListener {
    Disabled,
    Api,
    Dedicated(std::net::TcpListener),
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn serve(
    listener: std::net::TcpListener,
    bootstrap: BootstrapListener,
    senders: Senders,
    response_buf: usize,
    server_args: Arc<ServerArgs>,
    response_activity: ActivityCounter,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let state = Arc::new(AppState {
        senders,
        response_buf,
        server_args: server_args.clone(),
        chat_formatter,
        response_activity,
    });
    // Each endpoint module registers its own routes and merges here.
    let router = Router::new()
        .merge(common::routes())
        .merge(loads::routes())
        .merge(native_api::routes())
        .merge(openai::routes());

    // TODO(auth): no API-key boundary yet. Python gates every route (except
    // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
    // a configured `api_key` does NOT protect these routes.
    //
    // No body limit, matching the Python server.
    let mut app = router
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(state);

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<Arc<AppState>> above — and before `log::apply`, so bootstrap traffic
    // shows in the access log.
    let mut bootstrap_server = None;
    if !matches!(&bootstrap, BootstrapListener::Disabled) {
        let (routes, sweeper) = pd_bootstrap::router_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        match bootstrap {
            BootstrapListener::Api => {
                app = app.merge(routes);
                tracing::info!("PD KV bootstrap registry mounted on the api listener");
            }
            BootstrapListener::Dedicated(listener) => {
                tracing::info!(addr = ?listener.local_addr(), "PD KV bootstrap registry on dedicated listener");
                // Decode probes the dedicated bootstrap listener itself.
                let routes = routes.route("/health", axum::routing::get(|| async { "OK" }));
                bootstrap_server = Some((listener, log::apply(routes, &server_args)));
            }
            BootstrapListener::Disabled => unreachable!(),
        }
    }

    // Apply logging and access log middleware.
    let app = super::metrics::mount(app, server_args.metrics.clone());
    let app = log::apply(app, &server_args);

    // The listener was already bound synchronously in `runtime::start` (so a port
    // conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };
    // `with_connect_info` exposes the peer address to the access-log middleware.
    let serve = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    );
    tokio::select! {
        r = serve => {
            if let Err(e) = r {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        r = async move {
            match bootstrap_server {
                Some((listener, routes)) => {
                    let listener = tokio::net::TcpListener::from_std(listener)?;
                    axum::serve(
                        listener,
                        routes.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                    ).await
                }
                None => std::future::pending::<std::io::Result<()>>().await,
            }
        } => {
            if let Err(e) = r {
                tracing::error!(error = %e, "PD bootstrap server exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
    }
}
