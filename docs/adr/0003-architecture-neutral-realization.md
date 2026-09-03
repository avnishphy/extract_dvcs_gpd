# ADR 0003: Architecture-neutral pseudodata realization

Status: accepted, 2026-08-31.

Noise, covariance, nuisance draws, masks, observable inclusion, and seeds form
a realization independent of DeepSets/MAF tensors. Exact synthetic truth is a
sealed evaluation sidecar and is forbidden from the observation encoder.
