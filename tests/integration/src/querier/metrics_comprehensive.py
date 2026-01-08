from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import requests

from fixtures import types
from fixtures.auth import USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD
from fixtures.metrics import Metrics
from fixtures.metrics_generators import (
    generate_counter_with_resets,
    generate_non_monotonic_sum_series,
    generate_simple_counter_series,
    generate_simple_gauge_series,
    generate_sparse_series,
)
from src.querier.timeseries_utils import (
    find_named_result,
    index_series_by_label,
)

DEFAULT_STEP_INTERVAL = 60  # seconds
DEFAULT_TOLERANCE = 1e-9
QUERY_TIMEOUT = 30  # seconds

def make_query_request(
    signoz: types.SigNoz,
    token: str,
    start_ms: int,
    end_ms: int,
    queries: List[Dict],
    *,
    request_type: str = "time_series",
    format_options: Optional[Dict] = None,
    variables: Optional[Dict] = None,
    no_cache: bool = True,
    timeout: int = QUERY_TIMEOUT,
) -> requests.Response:
    if format_options is None:
        format_options = {"formatTableResultForUI": False, "fillGaps": False}

    payload = {
        "schemaVersion": "v1",
        "start": start_ms,
        "end": end_ms,
        "requestType": request_type,
        "compositeQuery": {"queries": queries},
        "formatOptions": format_options,
        "noCache": no_cache,
    }
    if variables:
        payload["variables"] = variables

    return requests.post(
        signoz.self.host_configs["8080"].get("/api/v5/query_range"),
        timeout=timeout,
        headers={"authorization": f"Bearer {token}"},
        json=payload,
    )


def build_builder_query(
    name: str,
    metric_name: str,
    time_aggregation: str,
    space_aggregation: str,
    *,
    temporality: str = "cumulative",
    step_interval: int = DEFAULT_STEP_INTERVAL,
    group_by: Optional[List[str]] = None,
    filter_expression: Optional[str] = None,
    functions: Optional[List[Dict]] = None,
    disabled: bool = False,
) -> Dict:
    spec: Dict[str, Any] = {
        "name": name,
        "signal": "metrics",
        "aggregations": [
            {
                "metricName": metric_name,
                "temporality": temporality,
                "timeAggregation": time_aggregation,
                "spaceAggregation": space_aggregation,
            }
        ],
        "stepInterval": step_interval,
        "disabled": disabled,
    }

    if group_by:
        spec["groupBy"] = [
            {
                "name": label,
            }
            for label in group_by
        ]

    if filter_expression:
        spec["filter"] = {"expression": filter_expression}

    if functions:
        spec["functions"] = functions

    return {"type": "builder_query", "spec": spec}


def build_formula_query(
    name: str,
    expression: str,
    *,
    functions: Optional[List[Dict]] = None,
    disabled: bool = False,
) -> Dict:
    spec: Dict[str, Any] = {
        "name": name,
        "expression": expression,
        "disabled": disabled,
    }
    if functions:
        spec["functions"] = functions
    return {"type": "builder_formula", "spec": spec}


def build_function(name: str, *args: Any) -> Dict:
    func: Dict[str, Any] = {"name": name}
    if args:
        func["args"] = [{"value": arg} for arg in args]
    return func


def get_series_values(response_json: Dict, query_name: str) -> List[Dict]:
    results = response_json.get("data", {}).get("data", {}).get("results", [])
    result = find_named_result(results, query_name)
    if not result:
        return []
    aggregations = result.get("aggregations", [])
    if not aggregations:
        return []
    # at the time of writing this, the series is always a list with one element
    series = aggregations[0].get("series", [])
    if not series:
        return []
    return series[0].get("values", [])


def get_all_series(response_json: Dict, query_name: str) -> List[Dict]:
    results = response_json.get("data", {}).get("data", {}).get("results", [])
    result = find_named_result(results, query_name)
    if not result:
        return []
    aggregations = result.get("aggregations", [])
    if not aggregations:
        return []
    # at the time of writing this, the series is always a list with one element
    return aggregations[0].get("series", [])


def get_scalar_value(response_json: Dict, query_name: str) -> Optional[float]:
    values = get_series_values(response_json, query_name)
    if values:
        return values[0].get("value")
    return None

def compare_values(
    v1: float,
    v2: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    return abs(v1 - v2) <= tolerance


def compare_series_values(
    values1: List[Dict],
    values2: List[Dict],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    if len(values1) != len(values2):
        return False

    sorted1 = sorted(values1, key=lambda x: x["timestamp"])
    sorted2 = sorted(values2, key=lambda x: x["timestamp"])

    for v1, v2 in zip(sorted1, sorted2):
        if v1["timestamp"] != v2["timestamp"]:
            return False
        if not compare_values(v1["value"], v2["value"], tolerance):
            return False
    return True

def compare_all_series(
    series1: List[Dict],
    series2: List[Dict],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    if len(series1) != len(series2):
        return False

    # oh my lovely python
    def series_key(s: Dict) -> str:
        labels = s.get("labels", [])
        return str(sorted([
            (lbl.get("key", {}).get("name", ""), lbl.get("value", ""))
            for lbl in labels
        ]))

    sorted1 = sorted(series1, key=series_key)
    sorted2 = sorted(series2, key=series_key)

    for s1, s2 in zip(sorted1, sorted2):
        if series_key(s1) != series_key(s2):
            return False
        if not compare_series_values(
            s1.get("values", []),
            s2.get("values", []),
            tolerance,
        ):
            return False
    return True


def assert_results_equal(
    result_cached: Dict,
    result_no_cache: Dict,
    query_name: str,
    context: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    values_cached = get_series_values(result_cached, query_name)
    values_no_cache = get_series_values(result_no_cache, query_name)

    sorted_cached = sorted(values_cached, key=lambda x: x["timestamp"])
    sorted_no_cache = sorted(values_no_cache, key=lambda x: x["timestamp"])

    assert len(sorted_cached) == len(sorted_no_cache), (
        f"{context}: Different number of values. "
        f"Cached: {len(sorted_cached)}, No-cache: {len(sorted_no_cache)}\n"
        f"Cached timestamps: {[v['timestamp'] for v in sorted_cached]}\n"
        f"No-cache timestamps: {[v['timestamp'] for v in sorted_no_cache]}"
    )

    for v_cached, v_no_cache in zip(sorted_cached, sorted_no_cache):
        assert v_cached["timestamp"] == v_no_cache["timestamp"], (
            f"{context}: Timestamp mismatch. "
            f"Cached: {v_cached['timestamp']}, No-cache: {v_no_cache['timestamp']}"
        )
        assert compare_values(v_cached["value"], v_no_cache["value"], tolerance), (
            f"{context}: Value mismatch at timestamp {v_cached['timestamp']}. "
            f"Cached: {v_cached['value']}, No-cache: {v_no_cache['value']}"
        )


def assert_all_series_equal(
    result_cached: Dict,
    result_no_cache: Dict,
    query_name: str,
    context: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    series_cached = get_all_series(result_cached, query_name)
    series_no_cache = get_all_series(result_no_cache, query_name)

    assert compare_all_series(series_cached, series_no_cache, tolerance), (
        f"{context}: Cached series differ from non-cached series"
    )

@pytest.fixture
def query_time_range() -> Callable[..., Tuple[datetime, int, int]]:
    # returns a function that generates a time range
    def _get_range(
        duration_minutes: int = 20,
        offset_minutes: int = 10,
    ) -> Tuple[datetime, int, int]:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        start_ms = int((now - timedelta(minutes=duration_minutes + offset_minutes)).timestamp() * 1000)
        end_ms = int((now - timedelta(minutes=offset_minutes)).timestamp() * 1000)
        return now, start_ms, end_ms

    return _get_range


@pytest.fixture
def auth_token(
    create_user_admin: None,
    get_token: Callable[[str, str], str],
) -> str:
    return get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)


class TestTimeAggregations:
    # test time aggregation functions across different metric types
    # time aggregations reduce multiple data points within a time bucket to a single value
    # valid combinations:
    # - gauge (unspecified): latest, sum, avg, min, max, count
    # - counter (cumulative/delta): rate, increase
    @pytest.mark.parametrize(
        "time_agg",
        ["sum", "avg", "min", "max", "count", "latest"],
    )
    def test_gauge_time_aggregations(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
        time_agg: str,
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = f"test_gauge_time_agg_{time_agg}"

        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            values,
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            time_agg,
            "sum",
            temporality="unspecified",
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"

        result_values = get_series_values(data, "A")
        assert len(result_values) > 0, f"No values returned for {time_agg}"

    @pytest.mark.parametrize(
        "time_agg,temporality",
        [
            ("rate", "cumulative"),
            ("rate", "delta"),
            ("increase", "cumulative"),
            ("increase", "delta"),
        ],
        ids=["rate_cumulative", "rate_delta", "increase_cumulative", "increase_delta"],
    )
    def test_counter_time_aggregations(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
        time_agg: str,
        temporality: str,
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = f"test_counter_{time_agg}_{temporality}"

        if temporality == "cumulative":
            # Cumulative: monotonically increasing values
            values = [float(i * 100) for i in range(10)]
        else:
            # Delta: per-interval values
            values = [100.0] * 10

        metrics = generate_simple_counter_series(
            metric_name,
            {"service": "test"},
            values,
            temporality=temporality.capitalize(),
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            time_agg,
            "sum",
            temporality=temporality,
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"
        assert len(get_series_values(data, "A")) > 0


class TestNonMonotonicSum:
    @pytest.mark.parametrize(
        "time_agg",
        ["sum", "avg", "min", "max", "count", "latest"],
    )
    def test_non_monotonic_sum_gauge_aggregations(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
        time_agg: str,
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = f"test_non_monotonic_sum_{time_agg}"

        # Values that go up and down (non-monotonic behavior)
        # Simulating queue depth: starts at 10, goes up to 50, back down to 20
        values = [10.0, 20.0, 35.0, 50.0, 45.0, 30.0, 25.0, 20.0, 25.0, 30.0]
        metrics = generate_non_monotonic_sum_series(
            metric_name,
            {"service": "test", "queue": "processing"},
            values,
            temporality="Cumulative",
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        # Non-monotonic Sums treated as gauges use "unspecified" temporality in queries
        query = build_builder_query(
            "A",
            metric_name,
            time_agg,
            "sum",
            temporality="unspecified",
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"

        result_values = get_series_values(data, "A")
        assert len(result_values) > 0, f"No values returned for {time_agg}"

    def test_non_monotonic_sum_with_fluctuating_values(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_non_monotonic_fluctuating"

        # Active connections pattern: goes up during load, down after
        values = [5.0, 15.0, 25.0, 40.0, 35.0, 20.0, 10.0, 5.0, 8.0, 12.0]
        metrics = generate_non_monotonic_sum_series(
            metric_name,
            {"service": "api", "metric_type": "active_connections"},
            values,
            temporality="Cumulative",
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        # Query for avg - should correctly average the fluctuating values
        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

    def test_non_monotonic_sum_min_max(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_non_monotonic_minmax"

        # Clear pattern: min=5, max=100
        values = [10.0, 5.0, 20.0, 100.0, 50.0, 30.0, 15.0, 10.0, 8.0, 12.0]
        metrics = generate_non_monotonic_sum_series(
            metric_name,
            {"service": "test"},
            values,
            temporality="Cumulative",
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        # Test min aggregation
        query_min = build_builder_query(
            "A",
            metric_name,
            "min",
            "sum",
            temporality="unspecified",
            step_interval=60,
        )

        response_min = make_query_request(signoz, auth_token, start_ms, end_ms, [query_min])
        assert response_min.status_code == HTTPStatus.OK

        data_min = response_min.json()
        result_values_min = get_series_values(data_min, "A")
        assert len(result_values_min) > 0

        # Test max aggregation
        query_max = build_builder_query(
            "A",
            metric_name,
            "max",
            "sum",
            temporality="unspecified",
            step_interval=60,
        )

        response_max = make_query_request(signoz, auth_token, start_ms, end_ms, [query_max])
        assert response_max.status_code == HTTPStatus.OK

        data_max = response_max.json()
        result_values_max = get_series_values(data_max, "A")
        assert len(result_values_max) > 0

    def test_non_monotonic_sum_delta_temporality(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_non_monotonic_delta"

        # Per-interval changes (can be negative)
        # Example: net change in queue depth per minute
        values = [5.0, 10.0, -3.0, 8.0, -5.0, -2.0, 4.0, 0.0, 3.0, -1.0]
        metrics = generate_non_monotonic_sum_series(
            metric_name,
            {"service": "queue-processor"},
            values,
            temporality="Delta",
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "sum",
            "sum",
            temporality="delta",
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        # Delta non-monotonic sums should return values
        assert len(result_values) > 0

    def test_non_monotonic_sum_multiple_series(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_non_monotonic_multi"

        metrics = []
        queues = ["high-priority", "low-priority", "batch"]
        for queue in queues:
            # Each queue has different baseline with fluctuations
            base = {"high-priority": 50, "low-priority": 20, "batch": 100}[queue]
            values = [float(base + (i % 10) - 5) for i in range(10)]
            metrics.extend(
                generate_non_monotonic_sum_series(
                    metric_name,
                    {"queue": queue},
                    values,
                    temporality="Cumulative",
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            group_by=["queue"],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        all_series = get_all_series(data, "A")
        assert len(all_series) == len(queues), (
            f"Expected {len(queues)} series, got {len(all_series)}"
        )

    def test_non_monotonic_sum_with_filter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_non_monotonic_filter"

        metrics = []
        for priority in ["high", "low"]:
            base = 100.0 if priority == "high" else 10.0
            values = [base + (i % 5) - 2 for i in range(10)]
            metrics.extend(
                generate_non_monotonic_sum_series(
                    metric_name,
                    {"priority": priority},
                    values,
                    temporality="Cumulative",
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            filter_expression="priority = 'high'",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

        # Filtered to high priority (base ~100), all values should be >= 90
        for v in result_values:
            assert v["value"] >= 90, "Filter should only include high priority data"


class TestSpaceAggregations:
    @pytest.mark.parametrize("space_agg", ["sum", "avg", "min", "max"])
    def test_space_aggregation_multiple_series(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
        space_agg: str,
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = f"test_space_agg_{space_agg}"

        # Create multiple series with different services
        metrics = []
        for i, service in enumerate(["frontend", "backend", "database"]):
            values = [float((i + 1) * 10 + j) for j in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"service": service},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            space_agg,
            temporality="unspecified",
            step_interval=60,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"
        assert len(get_series_values(data, "A")) > 0

    def test_space_aggregation_with_group_by(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_space_agg_group_by"

        metrics = []
        services = ["frontend", "backend"]
        for service in services:
            values = [float(10 + i) for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"service": service},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            group_by=["service"],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        all_series = get_all_series(data, "A")
        assert len(all_series) == len(services), (
            f"Expected {len(services)} series, got {len(all_series)}"
        )


class TestRateAndIncrease:

    def test_rate_cumulative_counter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_rate_cumulative"

        # Monotonically increasing counter
        values = [float(i * 100) for i in range(15)]
        metrics = generate_simple_counter_series(
            metric_name,
            {"service": "test"},
            values,
            temporality="Cumulative",
            start_time=now - timedelta(minutes=25),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "rate",
            "sum",
            temporality="cumulative",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

        # Rate should be approximately constant (100 per minute = 100/60 per second)
        for v in result_values:
            assert v["value"] >= 0, "Rate should not be negative"

    def test_increase_cumulative_counter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_increase_cumulative"

        values = [float(i * 50) for i in range(15)]
        metrics = generate_simple_counter_series(
            metric_name,
            {"service": "test"},
            values,
            temporality="Cumulative",
            start_time=now - timedelta(minutes=25),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "increase",
            "sum",
            temporality="cumulative",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

    def test_rate_handles_counter_reset(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_rate_reset"

        # Counter with reset at index 7
        metrics = generate_counter_with_resets(
            metric_name,
            {"service": "test"},
            num_points=15,
            rate_per_point=100.0,
            reset_at_indices=[7],
            start_time=now - timedelta(minutes=25),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "rate",
            "sum",
            temporality="cumulative",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")

        # Rate should remain non-negative even with resets
        for v in result_values:
            assert v["value"] >= 0, f"Rate should not be negative: {v['value']}"

    def test_delta_counter_sum(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_delta_sum"

        values = [50.0 + (i % 10) for i in range(15)]
        metrics = generate_simple_counter_series(
            metric_name,
            {"service": "test"},
            values,
            temporality="Delta",
            start_time=now - timedelta(minutes=25),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "sum",
            "sum",
            temporality="delta",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert len(get_series_values(data, "A")) > 0

class TestFiltersAndGrouping:

    def test_equality_filter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_filter_eq"

        metrics = []
        for status in ["200", "500"]:
            value = 100.0 if status == "200" else 10.0
            values = [value + i for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"status_code": status},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            filter_expression="status_code = '200'",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

        # All values should be from status_code=200 series (higher values)
        for v in result_values:
            assert v["value"] >= 100, "Filter should only include status_code=200"

    def test_not_equal_filter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_filter_neq"

        metrics = []
        for status in ["200", "500"]:
            value = 100.0 if status == "200" else 10.0
            values = [value + i for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"status_code": status},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            filter_expression="status_code != '200'",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

        # All values should be from status_code=500 series (lower values)
        for v in result_values:
            assert v["value"] < 100, "Filter should exclude status_code=200"

    def test_in_filter(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_filter_in"

        metrics = []
        for status in ["200", "201", "500"]:
            value = 100.0 if status in ["200", "201"] else 10.0
            values = [value + i for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"status_code": status},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            filter_expression="status_code IN ('200', '201')",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

    def test_group_by_single_dimension(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_group_by_single"

        metrics = []
        services = ["frontend", "backend", "database"]
        for service in services:
            values = [float(10 + i) for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"service": service},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            group_by=["service"],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        all_series = get_all_series(data, "A")
        assert len(all_series) == len(services)

    def test_group_by_multiple_dimensions(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_group_by_multi"

        metrics = []
        for service in ["frontend", "backend"]:
            for status in ["200", "500"]:
                values = [float(10 + i) for i in range(10)]
                metrics.extend(
                    generate_simple_gauge_series(
                        metric_name,
                        {"service": service, "status_code": status},
                        values,
                        start_time=now - timedelta(minutes=30),
                    )
                )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            group_by=["service", "status_code"],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        all_series = get_all_series(data, "A")
        # 2 services × 2 status codes = 4 series
        assert len(all_series) == 4

    def test_filter_with_group_by(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_filter_group_by"

        metrics = []
        for service in ["frontend", "backend"]:
            for status in ["200", "500"]:
                values = [float(10 + i) for i in range(10)]
                metrics.extend(
                    generate_simple_gauge_series(
                        metric_name,
                        {"service": service, "status_code": status},
                        values,
                        start_time=now - timedelta(minutes=30),
                    )
                )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            filter_expression="status_code = '200'",
            group_by=["service"],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        all_series = get_all_series(data, "A")
        # Filter to status_code=200, group by service = 2 series
        assert len(all_series) == 2

class TestFormulas:

    def test_simple_division(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_a = "test_formula_div_a"
        metric_b = "test_formula_div_b"

        metrics = generate_simple_gauge_series(
            metric_a,
            {"service": "test"},
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
            start_time=now - timedelta(minutes=30),
        )
        metrics.extend(
            generate_simple_gauge_series(
                metric_b,
                {"service": "test"},
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                start_time=now - timedelta(minutes=30),
            )
        )
        insert_metrics(metrics)

        queries = [
            build_builder_query("A", metric_a, "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("B", metric_b, "avg", "sum", temporality="unspecified", disabled=True),
            build_formula_query("F1", "A / B"),
        ]

        response = make_query_request(signoz, auth_token, start_ms, end_ms, queries)
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "F1")
        assert len(result_values) > 0

        # A / B should be 10 for all points
        for v in result_values:
            assert compare_values(v["value"], 10.0, tolerance=0.1)

    def test_percentage_formula(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_errors = "test_formula_pct_errors"
        metric_total = "test_formula_pct_total"

        # 10% error rate
        metrics = generate_simple_gauge_series(
            metric_errors,
            {"service": "test"},
            [10.0] * 10,
            start_time=now - timedelta(minutes=30),
        )
        metrics.extend(
            generate_simple_gauge_series(
                metric_total,
                {"service": "test"},
                [100.0] * 10,
                start_time=now - timedelta(minutes=30),
            )
        )
        insert_metrics(metrics)

        queries = [
            build_builder_query("A", metric_errors, "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("B", metric_total, "avg", "sum", temporality="unspecified", disabled=True),
            build_formula_query("F1", "(A / B) * 100"),
        ]

        response = make_query_request(signoz, auth_token, start_ms, end_ms, queries)
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "F1")
        assert len(result_values) > 0

        for v in result_values:
            assert compare_values(v["value"], 10.0, tolerance=0.1)

    def test_complex_formula(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()

        metric_names = ["test_formula_complex_a", "test_formula_complex_b",
                       "test_formula_complex_c", "test_formula_complex_d"]
        base_values = [10.0, 20.0, 100.0, 200.0]

        metrics = []
        for metric_name, base_val in zip(metric_names, base_values):
            values = [base_val] * 10
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"service": "test"},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        queries = [
            build_builder_query("A", metric_names[0], "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("B", metric_names[1], "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("C", metric_names[2], "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("D", metric_names[3], "avg", "sum", temporality="unspecified", disabled=True),
            build_formula_query("F1", "(A + B) / (C + D) * 100"),
        ]

        response = make_query_request(signoz, auth_token, start_ms, end_ms, queries)
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "F1")
        assert len(result_values) > 0

        # (10 + 20) / (100 + 200) * 100 = 30 / 300 * 100 = 10
        for v in result_values:
            assert compare_values(v["value"], 10.0, tolerance=0.1)


class TestFunctions:
    def test_fill_zero(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_fill_zero"

        # Create sparse data with gaps
        metrics = generate_sparse_series(
            metric_name,
            {"service": "test"},
            values_at_indices={0: 100.0, 5: 200.0, 9: 300.0},
            total_points=10,
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            functions=[build_function("fillZero")],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")

        # Should have values for all time points, with zeros for gaps
        zero_count = sum(1 for v in result_values if v["value"] == 0)
        assert zero_count > 0, "fillZero should add zero values for gaps"

    def test_cutoff_min(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_cutoff_min"

        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            values,
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            functions=[build_function("cutOffMin", 50.0)],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")

        # All values should be >= 50 after cutoff
        for v in result_values:
            assert v["value"] >= 50, f"cutOffMin should remove values < 50: {v['value']}"

    def test_clamp_max(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_clamp_max"

        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            values,
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            functions=[build_function("clampMax", 50.0)],
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")

        # All values should be <= 50 after clamp
        for v in result_values:
            assert v["value"] <= 50, f"clampMax should cap values at 50: {v['value']}"

class TestEdgeCases:

    def test_empty_result_set(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        _, start_ms, end_ms = query_time_range()

        query = build_builder_query(
            "A",
            "nonexistent_metric_12345",
            "avg",
            "sum",
            temporality="unspecified",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"

        result_values = get_series_values(data, "A")
        assert len(result_values) == 0

    def test_sparse_data(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_sparse"

        metrics = generate_sparse_series(
            metric_name,
            {"service": "test"},
            values_at_indices={0: 100.0, 9: 200.0},
            total_points=10,
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"

    def test_single_data_point(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_single_point"

        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            [100.0],
            start_time=now - timedelta(minutes=15),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert data["status"] == "success"


class TestCompositeQueries:

    def test_multiple_metrics_same_type(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()

        metrics = []
        for name in ["test_multi_a", "test_multi_b", "test_multi_c"]:
            values = [float(i * 10) for i in range(10)]
            metrics.extend(
                generate_simple_gauge_series(
                    name,
                    {"service": "test"},
                    values,
                    start_time=now - timedelta(minutes=30),
                )
            )
        insert_metrics(metrics)

        queries = [
            build_builder_query("A", "test_multi_a", "avg", "sum", temporality="unspecified"),
            build_builder_query("B", "test_multi_b", "avg", "sum", temporality="unspecified"),
            build_builder_query("C", "test_multi_c", "avg", "sum", temporality="unspecified"),
        ]

        response = make_query_request(signoz, auth_token, start_ms, end_ms, queries)
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        assert len(get_series_values(data, "A")) > 0
        assert len(get_series_values(data, "B")) > 0
        assert len(get_series_values(data, "C")) > 0

    def test_disabled_query_not_in_result(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
    ) -> None:
        now, start_ms, end_ms = query_time_range()
        metric_name = "test_disabled"

        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            [float(i) for i in range(10)],
            start_time=now - timedelta(minutes=30),
        )
        insert_metrics(metrics)

        queries = [
            build_builder_query("A", metric_name, "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("B", metric_name, "avg", "sum", temporality="unspecified", disabled=False),
        ]

        response = make_query_request(signoz, auth_token, start_ms, end_ms, queries)
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        # A is disabled, B should have values
        assert len(get_series_values(data, "A")) == 0
        assert len(get_series_values(data, "B")) > 0


class TestStepIntervals:
    @pytest.mark.parametrize("step", [60, 300, 900])
    def test_step_interval(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        insert_metrics: Callable[[List[Metrics]], None],
        query_time_range: Callable[..., Tuple[datetime, int, int]],
        step: int,
    ) -> None:
        now, start_ms, end_ms = query_time_range(duration_minutes=60, offset_minutes=10)
        metric_name = f"test_step_{step}"

        values = [float(i) for i in range(60)]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "test"},
            values,
            start_time=now - timedelta(minutes=70),
        )
        insert_metrics(metrics)

        query = build_builder_query(
            "A",
            metric_name,
            "avg",
            "sum",
            temporality="unspecified",
            step_interval=step,
        )

        response = make_query_request(signoz, auth_token, start_ms, end_ms, [query])
        assert response.status_code == HTTPStatus.OK

        data = response.json()
        result_values = get_series_values(data, "A")
        assert len(result_values) > 0

        # Verify timestamps are aligned to step
        for v in result_values:
            ts_ms = v["timestamp"]
            assert ts_ms % (step * 1000) == 0, f"Timestamp {ts_ms} not aligned to {step}s step"

@pytest.mark.cache
class TestCacheCorrectness:
    @pytest.mark.parametrize(
        "scenario,cache_range,query_range",
        [
            # (scenario_name, (cache_start_offset, cache_end_offset), (query_start_offset, query_end_offset))
            ("left_overlap", (30, 20), (40, 20)),      # Query extends left of cache
            ("right_overlap", (40, 30), (40, 20)),     # Query extends right of cache
            ("both_sides", (35, 25), (40, 20)),        # Query extends both sides
            ("complete_inside", (40, 20), (35, 25)),   # Query inside cached range
            ("disjoint", (50, 40), (30, 20)),          # No overlap
        ],
        ids=["left_overlap", "right_overlap", "both_sides", "complete_inside", "disjoint"],
    )
    def test_cache_overlap_scenarios(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
        scenario: str,
        cache_range: Tuple[int, int],
        query_range: Tuple[int, int],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = f"test_cache_{scenario}"

        # Create data for full range
        values = [float(i * 10) for i in range(50)]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": f"{scenario}-test"},
            values,
            start_time=now - timedelta(minutes=60),
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        query = build_builder_query(
            "A", metric_name, "avg", "sum", temporality="unspecified", step_interval=60
        )

        # Step 1: Populate cache
        cache_start = int((now - timedelta(minutes=cache_range[0])).timestamp() * 1000)
        cache_end = int((now - timedelta(minutes=cache_range[1])).timestamp() * 1000)
        response_initial = make_query_request(
            signoz, token, cache_start, cache_end, [query], no_cache=False
        )
        assert response_initial.status_code == HTTPStatus.OK

        # Step 2: Query with cache (may be partial hit)
        query_start = int((now - timedelta(minutes=query_range[0])).timestamp() * 1000)
        query_end = int((now - timedelta(minutes=query_range[1])).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=False
        )
        assert response_cached.status_code == HTTPStatus.OK

        # Step 3: Ground truth (no cache)
        response_no_cache = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=True
        )
        assert response_no_cache.status_code == HTTPStatus.OK

        # Compare
        assert_results_equal(
            response_cached.json(),
            response_no_cache.json(),
            "A",
            scenario,
        )

    def test_cache_multiple_non_contiguous_buckets(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = "test_cache_multi_bucket"

        values = [float(i * 10) for i in range(50)]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "multi-bucket-test"},
            values,
            start_time=now - timedelta(minutes=70),
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        query = build_builder_query(
            "A", metric_name, "avg", "sum", temporality="unspecified", step_interval=60
        )

        # Populate first bucket
        t1, t2 = (
            int((now - timedelta(minutes=60)).timestamp() * 1000),
            int((now - timedelta(minutes=50)).timestamp() * 1000),
        )
        make_query_request(signoz, token, t1, t2, [query], no_cache=False)

        # Populate second bucket (gap between t2 and t3)
        t3, t4 = (
            int((now - timedelta(minutes=40)).timestamp() * 1000),
            int((now - timedelta(minutes=30)).timestamp() * 1000),
        )
        make_query_request(signoz, token, t3, t4, [query], no_cache=False)

        # Query spanning all
        t5 = int((now - timedelta(minutes=20)).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, t1, t5, [query], no_cache=False
        )
        response_no_cache = make_query_request(
            signoz, token, t1, t5, [query], no_cache=True
        )

        assert_results_equal(
            response_cached.json(),
            response_no_cache.json(),
            "A",
            "multi_bucket",
        )

    def test_cache_with_filter(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = "test_cache_filter"

        metrics = []
        for status in ["200", "500"]:
            value_base = 100.0 if status == "200" else 10.0
            values = [value_base + i for i in range(40)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"status_code": status},
                    values,
                    start_time=now - timedelta(minutes=50),
                )
            )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        query = build_builder_query(
            "A", metric_name, "avg", "sum", temporality="unspecified",
            step_interval=60, filter_expression="status_code = '200'",
        )

        # Populate cache (partial range)
        cache_start = int((now - timedelta(minutes=35)).timestamp() * 1000)
        cache_end = int((now - timedelta(minutes=25)).timestamp() * 1000)
        make_query_request(signoz, token, cache_start, cache_end, [query], no_cache=False)

        # Query with both sides overlap
        query_start = int((now - timedelta(minutes=40)).timestamp() * 1000)
        query_end = int((now - timedelta(minutes=20)).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=False
        )
        response_no_cache = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=True
        )

        assert_results_equal(
            response_cached.json(),
            response_no_cache.json(),
            "A",
            "filter_partial_hit",
        )

    def test_cache_with_group_by(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = "test_cache_groupby"

        metrics = []
        for service in ["frontend", "backend", "database"]:
            value_base = {"frontend": 100.0, "backend": 200.0, "database": 300.0}[service]
            values = [value_base + i * 10 for i in range(40)]
            metrics.extend(
                generate_simple_gauge_series(
                    metric_name,
                    {"service": service},
                    values,
                    start_time=now - timedelta(minutes=50),
                )
            )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        query = build_builder_query(
            "A", metric_name, "avg", "sum", temporality="unspecified",
            step_interval=60, group_by=["service"],
        )

        # Populate cache (partial range)
        cache_start = int((now - timedelta(minutes=35)).timestamp() * 1000)
        cache_end = int((now - timedelta(minutes=25)).timestamp() * 1000)
        make_query_request(signoz, token, cache_start, cache_end, [query], no_cache=False)

        # Query with both sides overlap
        query_start = int((now - timedelta(minutes=40)).timestamp() * 1000)
        query_end = int((now - timedelta(minutes=20)).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=False
        )
        response_no_cache = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=True
        )

        assert_all_series_equal(
            response_cached.json(),
            response_no_cache.json(),
            "A",
            "groupby_partial_hit",
        )

    @pytest.mark.parametrize(
        "time_agg,temporality",
        [
            ("rate", "cumulative"),
            ("increase", "cumulative"),
            ("sum", "delta"),
        ],
        ids=["rate", "increase", "delta_sum"],
    )
    def test_cache_with_aggregation_types(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
        time_agg: str,
        temporality: str,
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = f"test_cache_{time_agg}_{temporality}"

        if temporality == "cumulative":
            values = [float(i * 100) for i in range(40)]
        else:
            values = [50.0 + (i % 10) for i in range(40)]

        metrics = generate_simple_counter_series(
            metric_name,
            {"service": f"{time_agg}-test"},
            values,
            temporality=temporality.capitalize(),
            start_time=now - timedelta(minutes=50),
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        query = build_builder_query(
            "A", metric_name, time_agg, "sum",
            temporality=temporality, step_interval=60,
        )

        # Populate cache (partial range)
        cache_start = int((now - timedelta(minutes=35)).timestamp() * 1000)
        cache_end = int((now - timedelta(minutes=25)).timestamp() * 1000)
        make_query_request(signoz, token, cache_start, cache_end, [query], no_cache=False)

        # Query with both sides overlap
        query_start = int((now - timedelta(minutes=40)).timestamp() * 1000)
        query_end = int((now - timedelta(minutes=20)).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=False
        )
        response_no_cache = make_query_request(
            signoz, token, query_start, query_end, [query], no_cache=True
        )

        assert_results_equal(
            response_cached.json(),
            response_no_cache.json(),
            "A",
            f"{time_agg}_{temporality}_partial_hit",
        )

    def test_cache_with_formula(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_a = "test_cache_formula_a"
        metric_b = "test_cache_formula_b"

        values_a = [100.0 + i * 10 for i in range(40)]
        values_b = [10.0 + i for i in range(40)]

        metrics = generate_simple_gauge_series(
            metric_a, {"service": "test"}, values_a,
            start_time=now - timedelta(minutes=50),
        )
        metrics.extend(
            generate_simple_gauge_series(
                metric_b, {"service": "test"}, values_b,
                start_time=now - timedelta(minutes=50),
            )
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        queries = [
            build_builder_query("A", metric_a, "avg", "sum", temporality="unspecified", disabled=True),
            build_builder_query("B", metric_b, "avg", "sum", temporality="unspecified", disabled=True),
            build_formula_query("F1", "A / B"),
        ]

        # Populate cache
        cache_start = int((now - timedelta(minutes=35)).timestamp() * 1000)
        cache_end = int((now - timedelta(minutes=25)).timestamp() * 1000)
        make_query_request(signoz, token, cache_start, cache_end, queries, no_cache=False)

        # Query with overlap
        query_start = int((now - timedelta(minutes=40)).timestamp() * 1000)
        query_end = int((now - timedelta(minutes=20)).timestamp() * 1000)

        response_cached = make_query_request(
            signoz, token, query_start, query_end, queries, no_cache=False
        )
        response_no_cache = make_query_request(
            signoz, token, query_start, query_end, queries, no_cache=True
        )

        assert_results_equal(
            response_cached.json(),
            response_no_cache.json(),
            "F1",
            "formula_partial_hit",
        )

    def test_cache_consistency_repeated_queries(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = "test_cache_consistency"

        values = [float(i * 10) for i in range(15)]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "consistency-test"},
            values,
            start_time=now - timedelta(minutes=25),
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        start_ms = int((now - timedelta(minutes=25)).timestamp() * 1000)
        end_ms = int((now - timedelta(minutes=10)).timestamp() * 1000)

        query = build_builder_query(
            "A", metric_name, "avg", "sum",
            temporality="unspecified", step_interval=60,
        )

        # Execute same query 3 times
        responses = []
        for _ in range(3):
            resp = make_query_request(signoz, token, start_ms, end_ms, [query], no_cache=False)
            assert resp.status_code == HTTPStatus.OK
            responses.append(resp.json())

        # All should match
        values1 = get_series_values(responses[0], "A")
        values2 = get_series_values(responses[1], "A")
        values3 = get_series_values(responses[2], "A")

        assert compare_series_values(values1, values2), "Query 1 and 2 differ"
        assert compare_series_values(values2, values3), "Query 2 and 3 differ"

    def test_cache_different_step_intervals(
        self,
        signoz: types.SigNoz,
        create_user_admin: None,
        get_token: Callable[[str, str], str],
        insert_metrics: Callable[[List[Metrics]], None],
    ) -> None:
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        metric_name = "test_cache_steps"

        values = [float(i) for i in range(30)]
        metrics = generate_simple_gauge_series(
            metric_name,
            {"service": "step-test"},
            values,
            start_time=now - timedelta(minutes=40),
        )
        insert_metrics(metrics)

        token = get_token(USER_ADMIN_EMAIL, USER_ADMIN_PASSWORD)
        start_ms = int((now - timedelta(minutes=40)).timestamp() * 1000)
        end_ms = int((now - timedelta(minutes=10)).timestamp() * 1000)

        for step in [60, 300]:
            query = build_builder_query(
                "A", metric_name, "avg", "sum",
                temporality="unspecified", step_interval=step,
            )

            response_no_cache = make_query_request(
                signoz, token, start_ms, end_ms, [query], no_cache=True
            )
            response_cached = make_query_request(
                signoz, token, start_ms, end_ms, [query], no_cache=False
            )

            values_no_cache = get_series_values(response_no_cache.json(), "A")
            values_cached = get_series_values(response_cached.json(), "A")

            assert compare_series_values(values_no_cache, values_cached), (
                f"Step {step}s: cached differs from non-cached"
            )

@pytest.mark.skip(reason="Histogram fixture support not yet implemented")
class TestHistogramQueries:
    @pytest.mark.parametrize("percentile", [50, 75, 90, 95, 99])
    def test_histogram_percentile(
        self,
        signoz: types.SigNoz,
        auth_token: str,
        percentile: int,
    ) -> None:
        # TODO(srikanthccv): add tests for exp hist
        pass
