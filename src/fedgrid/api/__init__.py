"""Observability layer: FastAPI backend + WebSocket stream feeding the React console.

:mod:`fedgrid.api.schemas` is the cross-language contract — every pydantic model
there is mirrored field-for-field by a TS type in ``frontend/src/types.ts``;
changing one side without the other is a contract violation.
"""

from __future__ import annotations
