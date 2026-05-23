"""Tests for CLI ��� verify commands work end-to-end."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess

PYTHON = sys.executable
MAIN = os.path.join(os.path.dirname(__file__), "..", "main.py")
SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, MAIN, *args],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )


# --- list ---

def test_list_command():
    r = _run_cli("list")
    assert r.returncode == 0
    assert "sma_cross" in r.stdout
    assert "rsi" in r.stdout


# --- run ---

def test_run_sma_cross():
    r = _run_cli("run", "sma_cross", SAMPLE, "--param", "fast=3", "--param", "slow=5")
    assert r.returncode == 0
    assert "Total Return" in r.stdout
    assert "Sharpe" in r.stdout


def test_run_rsi():
    r = _run_cli("run", "rsi", SAMPLE, "--param", "period=3", "--param", "oversold=30",
                 "--param", "overbought=70")
    assert r.returncode == 0
    assert "Total Return" in r.stdout


def test_run_json_output():
    r = _run_cli("run", "sma_cross", SAMPLE, "--param", "fast=3", "--param", "slow=5",
                 "--json")
    assert r.returncode == 0
    assert '"sharpe"' in r.stdout
    assert '"total_return"' in r.stdout


def test_run_unknown_strategy():
    r = _run_cli("run", "nonexistent", SAMPLE)
    assert r.returncode != 0


def test_run_custom_capital():
    r = _run_cli("run", "sma_cross", SAMPLE, "--param", "fast=3", "--param", "slow=5",
                 "--capital", "50000")
    assert r.returncode == 0
    assert "Total Return" in r.stdout


# --- optimize ---

def test_optimize_command():
    r = _run_cli("optimize", "sma_cross", SAMPLE,
                 "--grid", "fast=2,3,4", "--grid", "slow=4,5,6",
                 "--target", "sharpe", "--top", "2")
    assert r.returncode == 0
    assert "Best params" in r.stdout
    assert "Best sharpe" in r.stdout
    assert "9 combinations" in r.stdout


# --- montecarlo ---

def test_montecarlo_command():
    r = _run_cli("montecarlo", "sma_cross", SAMPLE,
                 "--param", "fast=3", "--param", "slow=5",
                 "--paths", "50", "--seed", "42")
    assert r.returncode == 0
    assert "Monte Carlo" in r.stdout
    assert "Prob of Ruin" in r.stdout


def test_montecarlo_bootstrap_method():
    r = _run_cli("montecarlo", "sma_cross", SAMPLE,
                 "--param", "fast=3", "--param", "slow=5",
                 "--paths", "30", "--method", "bootstrap", "--seed", "1")
    assert r.returncode == 0
    assert "Monte Carlo" in r.stdout


def test_montecarlo_json():
    r = _run_cli("montecarlo", "sma_cross", SAMPLE,
                 "--param", "fast=3", "--param", "slow=5",
                 "--paths", "30", "--seed", "7", "--json")
    assert r.returncode == 0
    assert '"prob_ruin"' in r.stdout


# --- help ---

def test_help():
    r = _run_cli("--help")
    assert r.returncode == 0
    assert "run" in r.stdout
    assert "optimize" in r.stdout
    assert "list" in r.stdout
