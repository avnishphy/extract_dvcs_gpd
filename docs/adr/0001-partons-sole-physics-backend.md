# ADR 0001: PARTONS is the sole physics backend

Status: accepted, 2026-08-31.

PARTONS/C++ owns GPD evaluation, evolution, CFF convolution, native moments,
DVCS/BH/interference observables, and exact posterior reevaluation. Python may
operate on saved native quantities but must not supply a fallback forward
model. Unsupported native interfaces fail closed.
