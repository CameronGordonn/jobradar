-- New postings per week by role family and company (deduplicated), last 52 weeks.
select
    date_trunc('week', posted_date)::date as week_start,
    role_family,
    company,
    date_trunc('week', posted_date)::date::varchar || '|' || role_family || '|' || company as grain_key,
    count(distinct cluster_id) as new_postings,
    count(distinct cluster_id) filter (where is_open) as still_open
from {{ ref('int_jobs') }}
where posted_date >= current_date - 364
group by 1, 2, 3, 4
