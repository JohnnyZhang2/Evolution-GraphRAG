from scripts.check_requirements_imports import check_imports


def test_required_packages_importable():
    report = check_imports()
    missing_required = [r.name for r in report.results if r.category == 'required' and not r.importable]
    assert not missing_required, f"Missing required imports: {missing_required}"  # Provide detail


def test_report_counts_consistency():
    report = check_imports()
    # Validate counts line up with individual results
    calc_req_missing = sum(1 for r in report.results if r.category == 'required' and not r.importable)
    calc_opt_missing = sum(1 for r in report.results if r.category == 'optional' and not r.importable)
    assert calc_req_missing == report.required_missing
    assert calc_opt_missing == report.optional_missing
