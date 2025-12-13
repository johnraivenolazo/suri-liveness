"""Domain entities: LabelSpec and related logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

LabelValue = Union[int, str]


@dataclass(frozen=True)
class LabelSpec:
    """Defines how to obtain a class id from a metadata row.

    The training code expects a class id in [0, num_classes-1].

    Two common cases:
    - direct: the metadata already stores a class id
    - mapped: the metadata stores a raw code that needs mapping to a class id
    """

    label_column: Union[int, str]
    raw_to_class: Optional[Mapping[LabelValue, int]] = None

    def to_class_id(self, raw_value: Any) -> int:
        if self.raw_to_class is None:
            try:
                return int(raw_value)
            except Exception as e:  # pragma: no cover
                raise ValueError(
                    f"Could not convert label value to int: {raw_value!r}"
                ) from e

        key: LabelValue
        if isinstance(raw_value, (int, str)):
            key = raw_value
        else:
            key = str(raw_value)

        if key in self.raw_to_class:
            return int(self.raw_to_class[key])

        # Try int conversion if mapping keys are ints/strings.
        try:
            key_int = int(key)  # type: ignore[arg-type]
        except Exception:
            key_int = None
        if key_int is not None and key_int in self.raw_to_class:
            return int(self.raw_to_class[key_int])

        raise ValueError(f"Unmapped label value: {raw_value!r}")


def infer_label_spec(
    columns: Sequence[Any], *, preferred: Optional[Union[int, str]] = None
) -> LabelSpec:
    """Infer a reasonable LabelSpec from a dataframe's columns.

    - If preferred is provided and present, use it.
    - Else, prefer common string columns.
    - Else, fall back to a common integer column if present.
    """

    if preferred is not None and preferred in columns:
        return LabelSpec(label_column=preferred)

    for name in ("label", "class", "target", "y"):
        if name in columns:
            return LabelSpec(label_column=name)

    # A common legacy format stores a raw type code under an integer column.
    if 40 in columns:
        # Default mapping groups raw codes into 3 classes.
        # This is only used when no explicit label column is present.
        raw_to_class = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            7: 2,
            8: 2,
            9: 2,
        }
        return LabelSpec(label_column=40, raw_to_class=raw_to_class)

    raise ValueError(
        "Could not infer label column. Provide --label-col and optionally --label-map-json."
    )
