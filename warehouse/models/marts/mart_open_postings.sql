-- Postings open now, one row per near-duplicate cluster.
with open_jobs as (
    select
        *,
        count(*) over (partition by cluster_id) as variants,
        row_number() over (partition by cluster_id order by posted_at nulls last, job_id) as rn
    from {{ ref('int_jobs') }}
    where is_open
)

select
    cluster_id as posting_id,
    job_id,
    source,
    company,
    title_clean as title,
    location,
    url,
    role_family,
    workplace_type,
    is_remote,
    posted_date,
    salary_min_annual,
    salary_max_annual,
    salary_currency,
    variants
from open_jobs
where rn = 1
