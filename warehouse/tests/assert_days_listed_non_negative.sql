-- A posting can't be last seen before it was published.
select job_id from {{ ref('mart_posting_lifecycle') }} where days_listed < 0
