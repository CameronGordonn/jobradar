-- Board postings that have closed, and how long each was listed: from the date it was
-- published to the last run that still saw it. USAJOBS is left out because its searches
-- only cover the last 14 days, so "last seen" there is not when the posting closed. Postings
-- from boards that stopped loading are left out too: they disappeared with the board.
select
    job_id,
    source,
    company,
    role_family,
    posted_date,
    first_ingested_at::date as first_seen_date,
    ingested_at::date as last_seen_date,
    datediff('day', posted_date, ingested_at::date) as days_listed
from {{ ref('int_jobs') }}
where
    source != 'usajobs'
    and board_active
    and not is_open
    and posted_at is not null
