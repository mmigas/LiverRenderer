/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "mitsuba/render/ior.h"

#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>


NAMESPACE_BEGIN(mitsuba)
    enum EBioType {
        EAbsorber,
        EAttenuator,
        EAbsorberAndAttenuator,
        ENULL
    };

    struct OrganicMaterialEntry {
        const char* name;
        EBioType type;
        const char* filepath;
    };

    static OrganicMaterialEntry organicMaterialData[] = {
        /*Glisson Capsule*/
        {"Collagen", EAttenuator, "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/capsule/collagen.py"},
        {"Elastin", EAttenuator, "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/capsule/elastin.py"},

        /*Liver Parenchyma*/
        {"Blood", EAbsorber, "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/parenchyma/blood.py"},
        {"WaterLipid", EAbsorber, "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/parenchyma/water_lipid.py"},
        {"Bile", EAbsorber, "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/parenchyma/bile.py"},
        {
            "Hepatocity", EAbsorberAndAttenuator,
            "/mnt/c/dev/my-mitsuba-master/dist/plugins/liver/parenchyma/hepatocity.py"
        },

        {NULL, ENULL, ""}
    };


NAMESPACE_END(mitsuba)
