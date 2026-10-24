-- Every clustered job id exists in int_jobs (relationships test on a source table).
select c.job_id
from {{ source('jobradar', 'job_clusters') }} as c
left join {{ ref('int_jobs') }} as j using (job_id)
where j.job_id is null
