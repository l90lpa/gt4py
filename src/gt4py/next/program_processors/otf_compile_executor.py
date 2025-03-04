# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from typing import Any, Generic, Optional, TypeVar

from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, recipes, stages, workflow
from gt4py.next.program_processors import processor_interface as ppi


SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
LS = TypeVar("LS", bound=languages.LanguageSettings)
HashT = TypeVar("HashT")


@dataclasses.dataclass(frozen=True)
class OTFCompileExecutor(ppi.ProgramExecutor, Generic[SrcL, LS, TgtL, HashT]):
    otf_workflow: recipes.OTFCompileWorkflow[SrcL, LS, TgtL]
    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args, **kwargs: Any) -> None:
        self.otf_workflow(stages.ProgramCall(program, args, kwargs))(
            *args, offset_provider=kwargs["offset_provider"]
        )

    @property
    def __name__(self) -> str:
        return self.name or repr(self)


@dataclasses.dataclass(frozen=True)
class CachedOTFCompileExecutor(ppi.ProgramExecutor, Generic[SrcL, LS, TgtL, HashT]):
    otf_workflow: workflow.CachedStep[stages.ProgramCall, stages.CompiledProgram, HashT]
    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args, **kwargs: Any) -> None:
        self.otf_workflow(stages.ProgramCall(program, args, kwargs))(
            *args, offset_provider=kwargs["offset_provider"]
        )

    @property
    def __name__(self) -> str:
        return self.name or repr(self)
