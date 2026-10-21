/**
 * CHARGEUP - Client-Side Interactive Logic
 * Handles:
 * 1. Live prediction API requests via fetch to FastAPI POST /predict
 * 2. Preset scenario switching (Rush hour, morning commute, weekend, off-peak)
 * 3. Quick calculator on the overview dashboard
 * 4. Interactive table filtering (Fault table & Station clusters table)
 */

document.addEventListener("DOMContentLoaded", function () {
    // Re-initialize Lucide icons if available
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // -------------------------------------------------------------
    // 1. QUICK CALCULATOR ON OVERVIEW DASHBOARD
    // -------------------------------------------------------------
    const quickBtn = document.getElementById("btn-quick-predict");
    if (quickBtn) {
        quickBtn.addEventListener("click", async function () {
            const hour = parseInt(document.getElementById("quick-hour").value, 10);
            const dayOfWeek = parseInt(document.getElementById("quick-day").value, 10);
            const prevHour = parseFloat(document.getElementById("quick-prev-hour").value);
            const prevDay = parseFloat(document.getElementById("quick-prev-day").value);
            const isWeekend = (dayOfWeek === 5 || dayOfWeek === 6) ? 1 : 0;
            const month = 10; // Default current month

            quickBtn.disabled = true;
            quickBtn.innerHTML = '<i data-lucide="loader" class="spin-icon"></i> Calculating...';
            if (window.lucide) window.lucide.createIcons();

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        hour: hour,
                        day_of_week: dayOfWeek,
                        month: month,
                        is_weekend: isWeekend,
                        prev_hour_demand: prevHour,
                        prev_day_demand: prevDay
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }

                const data = await response.json();
                const predVal = data.predicted_demand_kwh.toFixed(2);
                
                const resultBox = document.getElementById("quick-result-box");
                const predDisplay = document.getElementById("quick-pred-val");
                if (resultBox && predDisplay) {
                    predDisplay.textContent = predVal;
                    resultBox.style.display = "block";
                }
            } catch (err) {
                alert("Error calling prediction API: " + err.message);
            } finally {
                quickBtn.disabled = false;
                quickBtn.innerHTML = '<i data-lucide="zap"></i> Calculate Next-Hour Demand';
                if (window.lucide) window.lucide.createIcons();
            }
        });
    }

    // -------------------------------------------------------------
    // 2. DEMAND FORECAST WORKBENCH (/forecast)
    // -------------------------------------------------------------
    const forecastForm = document.getElementById("forecast-form");
    const presetButtons = document.querySelectorAll(".btn-preset");
    
    // Preset configurations matching pipeline samples
    const presets = {
        weekday_peak: {
            hour: 18,
            day_of_week: 0,
            month: 10,
            is_weekend: 0,
            prev_hour_demand: 45.5,
            prev_day_demand: 42.0
        },
        weekday_morning: {
            hour: 8,
            day_of_week: 2,
            month: 6,
            is_weekend: 0,
            prev_hour_demand: 28.0,
            prev_day_demand: 26.5
        },
        weekend_afternoon: {
            hour: 14,
            day_of_week: 5,
            month: 8,
            is_weekend: 1,
            prev_hour_demand: 35.0,
            prev_day_demand: 31.2
        },
        night_low: {
            hour: 3,
            day_of_week: 3,
            month: 11,
            is_weekend: 0,
            prev_hour_demand: 4.2,
            prev_day_demand: 5.0
        }
    };

    if (presetButtons.length > 0) {
        presetButtons.forEach(btn => {
            btn.addEventListener("click", function () {
                presetButtons.forEach(b => b.classList.remove("active"));
                this.classList.add("active");

                const presetKey = this.getAttribute("data-preset");
                const cfg = presets[presetKey];
                if (cfg) {
                    document.getElementById("input-hour").value = cfg.hour;
                    document.getElementById("input-day").value = cfg.day_of_week;
                    document.getElementById("input-month").value = cfg.month;
                    document.getElementById("input-weekend").value = cfg.is_weekend;
                    document.getElementById("input-prev-hour").value = cfg.prev_hour_demand;
                    document.getElementById("input-prev-day").value = cfg.prev_day_demand;
                    
                    // Trigger forecast calculation automatically
                    executeForecast();
                }
            });
        });
    }

    if (forecastForm) {
        forecastForm.addEventListener("submit", function (e) {
            e.preventDefault();
            executeForecast();
        });
    }

    const resetBtn = document.getElementById("btn-reset-form");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            if (forecastForm) forecastForm.reset();
            const firstPreset = document.querySelector(".btn-preset");
            if (firstPreset) firstPreset.click();
        });
    }

    async function executeForecast() {
        const hourInput = document.getElementById("input-hour");
        if (!hourInput) return;

        const payload = {
            hour: parseInt(document.getElementById("input-hour").value, 10),
            day_of_week: parseInt(document.getElementById("input-day").value, 10),
            month: parseInt(document.getElementById("input-month").value, 10),
            is_weekend: parseInt(document.getElementById("input-weekend").value, 10),
            prev_hour_demand: parseFloat(document.getElementById("input-prev-hour").value),
            prev_day_demand: parseFloat(document.getElementById("input-prev-day").value)
        };

        const submitBtn = document.getElementById("btn-submit-forecast");
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i data-lucide="loader" class="spin-icon"></i> Forecasting...';
            if (window.lucide) window.lucide.createIcons();
        }

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || `HTTP ${resp.status}`);
            }

            const res = await resp.json();
            const val = res.predicted_demand_kwh;
            
            // Update display
            const valEl = document.getElementById("forecast-output-val");
            if (valEl) valEl.textContent = val.toFixed(2);

            // Update Level Pill
            const pill = document.getElementById("demand-level-pill");
            if (pill) {
                if (val > 40) {
                    pill.className = "level-pill level-high";
                    pill.innerHTML = '<i data-lucide="zap"></i> High Demand Period (> 40 kWh)';
                } else if (val > 15) {
                    pill.className = "level-pill level-med";
                    pill.innerHTML = '<i data-lucide="activity"></i> Moderate Demand Period (15 - 40 kWh)';
                } else {
                    pill.className = "level-pill level-low";
                    pill.innerHTML = '<i data-lucide="moon"></i> Low / Off-Peak Demand (< 15 kWh)';
                }
            }

            // Update JSON debug box
            const jsonBox = document.getElementById("api-response-json");
            if (jsonBox) {
                jsonBox.textContent = JSON.stringify(res, null, 2);
            }
        } catch (err) {
            alert("Prediction error: " + err.message);
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i data-lucide="play"></i> Predict Next-Hour Demand';
                if (window.lucide) window.lucide.createIcons();
            }
        }
    }

    // -------------------------------------------------------------
    // 3. FAULT DETECTION TABLE FILTER (/fault-detection)
    // -------------------------------------------------------------
    const stationFilter = document.getElementById("filter-station-input");
    if (stationFilter) {
        stationFilter.addEventListener("input", function () {
            const query = this.value.toLowerCase().trim();
            const rows = document.querySelectorAll("#fault-table tbody tr");
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                if (text.includes(query)) {
                    row.style.display = "";
                } else {
                    row.style.display = "none";
                }
            });
        });
    }

    // -------------------------------------------------------------
    // 4. STATION SEGMENTATION TABLE FILTER (/segmentation)
    // -------------------------------------------------------------
    const clusterSelect = document.getElementById("cluster-filter-select");
    const stationSearch = document.getElementById("station-search-box");

    function filterStations() {
        const clusterVal = clusterSelect ? clusterSelect.value : "all";
        const searchVal = stationSearch ? stationSearch.value.toLowerCase().trim() : "";
        const rows = document.querySelectorAll("#stations-table tbody tr");

        rows.forEach(row => {
            const cluster = row.getAttribute("data-cluster") || "";
            const text = row.textContent.toLowerCase();

            const matchesCluster = (clusterVal === "all" || cluster === clusterVal);
            const matchesSearch = (!searchVal || text.includes(searchVal));

            if (matchesCluster && matchesSearch) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        });
    }

    if (clusterSelect) clusterSelect.addEventListener("change", filterStations);
    if (stationSearch) stationSearch.addEventListener("input", filterStations);
});
