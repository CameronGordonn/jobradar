-- One row per posting from every source, cleaned and classified.
with jobs as (
    select
        *,
        trim(regexp_replace(title, '\s+', ' ', 'g')) as title_clean,
        {{ normalize_title('title') }} as title_norm
    from {{ ref('int_jobs_unioned') }}
),

family as (
    select
        j.job_id,
        r.role_family
    from jobs as j
    inner join {{ ref('role_families') }} as r on regexp_matches(j.title_norm, r.pattern)
    qualify row_number() over (partition by j.job_id order by r.priority) = 1
),

latest_load as (
    -- A board posting is open while it keeps showing up in that board's latest load, and the
    -- board itself is still being loaded: a board dropped from sources.yaml, or one that 404s
    -- for good, stops loading, and its postings would otherwise stay open forever.
    select
        source,
        board,
        max(ingested_at) as last_load,
        max(ingested_at) >= max(max(ingested_at)) over (partition by source) - interval 3 day as board_active
    from jobs
    group by 1, 2
),

clusters as (
    select
        job_id,
        cluster_id
    from {{ source('jobradar', 'job_clusters') }}
)

select
    j.job_id,
    j.source,
    j.source_job_id,
    j.board,
    trim(j.company) as company,
    j.title,
    j.title_clean,
    j.title_norm,
    j.location,
    j.url,
    j.department,
    j.employment_type,
    j.workplace_type,
    coalesce(j.remote_flag, false)
    or coalesce(j.workplace_type = 'remote', false)
    or regexp_matches(lower(coalesce(j.location, '')), '\bremote\b') as is_remote,
    coalesce(f.role_family, 'other') as role_family,
    j.posted_at,
    coalesce(j.posted_at, j.first_ingested_at)::date as posted_date,
    j.closes_at,
    j.salary_min,
    j.salary_max,
    j.salary_currency,
    j.salary_interval,
    {{ annualize('j.salary_min', 'j.salary_interval') }} as salary_min_annual,
    {{ annualize('j.salary_max', 'j.salary_interval') }} as salary_max_annual,
    j.first_ingested_at,
    j.ingested_at,
    l.board_active,
    case
        -- USAJOBS searches only return recent announcements, so use the closing date instead.
        when j.source = 'usajobs' then coalesce(j.closes_at::date >= current_date, j.ingested_at = l.last_load)
        else j.ingested_at = l.last_load and l.board_active
    end as is_open,
    md5(lower(trim(j.company)) || '|' || j.title_norm || '|' || lower(coalesce(j.location, ''))) as exact_key,
    -- Near-duplicate cluster when the embedding dedupe has run, else exact company + title + location.
    coalesce(
        c.cluster_id,
        md5(lower(trim(j.company)) || '|' || j.title_norm || '|' || lower(coalesce(j.location, '')))
    ) as cluster_id
from jobs as j
left join family as f using (job_id)
left join latest_load as l on j.source = l.source and l.board is not distinct from j.board
left join clusters as c using (job_id)
