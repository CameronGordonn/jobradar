-- Per source per day: postings first published that day (raw and deduplicated), and how many
-- postings the source returned in that day's last run.
with jobs as (
    select * from {{ ref('int_jobs') }}
),

bounds as (
    -- Run times are UTC and current_date follows the session time zone, so end at whichever is later.
    select
        greatest(min(posted_date), current_date - 365) as first_day,
        greatest(
            current_date,
            (select max(ir.started_at)::date from {{ source('jobradar', 'ingest_runs') }} as ir)
        ) as last_day
    from jobs
),

days as (
    select
        unnest(generate_series((select first_day from bounds), (select last_day from bounds), interval 1 day))::date
            as day
),

sources as (
    select distinct source from jobs
),

new_postings as (
    select
        posted_date as day,
        source,
        count(*) as new_postings,
        count(distinct cluster_id) as new_postings_deduped
    from jobs
    group by 1, 2
),

runs as (
    select
        started_at::date as day,
        source,
        arg_max(rows_fetched, started_at) as postings_in_feed
    from {{ source('jobradar', 'ingest_runs') }}
    group by 1, 2
)

select
    d.day,
    s.source,
    d.day::varchar || '|' || s.source as day_source,
    coalesce(n.new_postings, 0) as new_postings,
    coalesce(n.new_postings_deduped, 0) as new_postings_deduped,
    r.postings_in_feed
from days as d
cross join sources as s
left join new_postings as n on d.day = n.day and s.source = n.source
left join runs as r on d.day = r.day and s.source = r.source
