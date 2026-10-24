-- Every posting in the open mart must be open in int_jobs.
select m.job_id
from {{ ref('mart_open_postings') }} as m
inner join {{ ref('int_jobs') }} as j using (job_id)
where not j.is_open
