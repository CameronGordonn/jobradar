-- Every posting in the open mart must be open in int_jobs.
select m.job_id
from {{ ref('mart_open_postings') }} m
join {{ ref('int_jobs') }} j using (job_id)
where not j.is_open
