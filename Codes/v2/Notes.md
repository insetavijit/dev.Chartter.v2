THE GOAL

Then innisial goal was to create multiple tables for better data handling
and filterring like

Chart:1D -- 3Y range <1008>
Chart:4H -- 1Y range <1440> [6*5*4*12]

Chart:1H -- 1M range <672> [24*7*4]
--
Chart:15MIN - 1ST Week | Chart:15MIN - 1ST Week <480> [4*24*5] or 672 [(4*24*7)] * 2
Chart:15MIN - 3ED Week | Chart:15MIN - 4th Week <480> [4*24*5] or 672 [(4*24*7)] * 2
--
Chart:15MIN - Currunt week <480> [4*24*5] or 672 [(4*24*7)] * 2
Chart:5MIN - 1ST DAY | Chart:1MIN - 2ND DAY <288> [60/5*24]
Chart:5MIN - 3ND DAY | Chart:1MIN - 4ND DAY <288> [60/5*24]
Chart:1MIN - Current Day <1440> [60*24]
---
trading Chart:
    | 01H
1M  +------
    | 15M
