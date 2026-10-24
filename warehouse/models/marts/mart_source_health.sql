-- Each run's row count per source next to the quality check, for the dashboard and alerting.
select
    r.run_id,
    r.source,
    r.started_at,
    r.rows_fetched,
    r.rows_new,
    q.trailing_runs,
    q.trailing_median,
    q.ratio,
    coalesce(q.flagged, false) as flagged,
    q.reason
from {{ source('jobradar', 'ingest_runs') }} r
left join {{ source('jobradar', 'quality_checks') }} q using (run_id, source)
