import json

from jobradar import dashboard, db

SOURCES = """
sources:
  usajobs: {enabled: true, keywords: [a, b], locations: [California, null]}
  greenhouse: {enabled: true, boards: [{token: x, name: X}, {token: y, name: Y}]}
  lever: {enabled: true, boards: []}
  ashby: {enabled: true, boards: [{token: z, name: Z}]}
"""


def test_dashboard_exports_aggregates_only(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    con.execute("CREATE SCHEMA marts")
    con.execute("""CREATE TABLE marts.mart_open_postings AS SELECT * FROM (VALUES
        ('p1', 'greenhouse:1', 'greenhouse', 'X', 'Secret Title One', 'SF', 'https://x/1', 'machine_learning', true, 100000.0, 140000.0, 'USD', 1),
        ('p2', 'greenhouse:2', 'greenhouse', 'X', 'Secret Title Two', 'SF', 'https://x/2', 'data_science', false, NULL, NULL, NULL, 2),
        ('p3', 'ashby:3', 'ashby', 'Z', 'Secret Title Three', 'NYC', 'https://z/3', 'machine_learning', false, 90000.0, 110000.0, 'USD', 1))
        AS t(posting_id, job_id, source, company, title, location, url, role_family, is_remote,
             salary_min_annual, salary_max_annual, salary_currency, variants)""")
    con.execute("""CREATE TABLE marts.mart_daily_postings AS SELECT * FROM (VALUES
        (current_date - 10, 'greenhouse', 3, 2, NULL::INTEGER), (current_date, 'greenhouse', 0, 0, 40))
        AS t(day, source, new_postings, new_postings_deduped, postings_in_feed)""")
    con.execute("""CREATE TABLE marts.mart_source_health AS SELECT * FROM (VALUES
        ('r1', 'greenhouse', TIMESTAMP '2026-10-01 00:00:00', 40, 41.0, false, 'ok'))
        AS t(run_id, source, started_at, rows_fetched, trailing_median, flagged, reason)""")
    con.execute("""CREATE TABLE marts.mart_posting_lifecycle AS SELECT * FROM (VALUES
        ('greenhouse:9', 'greenhouse', 'X', 'machine_learning', 12)) AS t(job_id, source, company, role_family, days_listed)""")
    con.execute("""CREATE TABLE marts.mart_posting_changes AS SELECT * FROM (VALUES
        ('greenhouse:1', 2, 1, 0, 0, 0), ('greenhouse:2', 1, 0, 0, 0, 0))
        AS t(job_id, versions, title_changes, location_changes, salary_changes, workplace_changes)""")
    (tmp_path / "sources.yaml").write_text(SOURCES)
    d = dashboard.build(con, tmp_path / "sources.yaml", tmp_path / "site")
    assert d["tiles"]["open_postings"] == 3 and d["tiles"]["open_before_dedupe"] == 4
    assert d["closed_postings"] == 1 and d["lifecycle_by_family"] == []   # under 10 per family: not shown
    assert d["changes"] == {"tracked": 2, "edited": 1, "salary_edited": 0}
    assert d["scope"] == {"usajobs_queries": 4, "usajobs_keywords": ["a", "b"],
                          "boards": {"greenhouse": 2, "lever": 0, "ashby": 1}, "board_count": 3}
    assert d["feed_by_source"] == [{"x": d["feed_by_source"][0]["x"], "source": "greenhouse", "n": 40}]
    text = (tmp_path / "site/data.json").read_text()
    assert "Secret Title" not in text and "https://x/" not in text  # no individual postings
    assert (tmp_path / "site/index.html").exists()
    json.loads(text)
