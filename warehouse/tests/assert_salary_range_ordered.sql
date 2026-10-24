-- A salary range never has its minimum above its maximum.
select job_id from {{ ref('int_jobs') }}
where salary_min_annual > salary_max_annual
