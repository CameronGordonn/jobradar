-- The snapshot keeps exactly one open-ended (current) version per posting.
select job_id from {{ ref('snap_postings') }}
where dbt_valid_to is null
group by job_id having count(*) > 1
