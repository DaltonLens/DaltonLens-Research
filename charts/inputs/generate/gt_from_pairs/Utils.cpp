//
// Copyright (c) 2017, Nicolas Burrus
// This software may be modified and distributed under the terms
// of the BSD license.  See the LICENSE file for details.
//

#include "Utils.h"

#include <chrono>
#include <cstdarg>
#include <iostream>

namespace dl
{
    std::string formatted (const char* fmt, ...)
    {
        char buf [2048];
        buf[2047] = '\0';
        va_list args;
        va_start(args, fmt);
        vsnprintf (buf, 2047, fmt, args);
        va_end (args);
        return buf;
    }
}

namespace dl
{

    double currentDateInSeconds ()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    void ScopeTimer :: start ()
    {
        _startTime = currentDateInSeconds();
    }
    
    void ScopeTimer :: stop ()
    {
        const auto endTime = currentDateInSeconds();
        const auto deltaTime = endTime - _startTime;
        
        std::cerr << dl::formatted("[TIME] elasped in %s: %.1f ms", _label.c_str(), deltaTime*1e3);
    }

} // dl
