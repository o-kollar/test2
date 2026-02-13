const https = require('https');

// Simple Weather Tool — Open-Meteo API (no key required)
// Usage: node weather_api.js [city] [+day|rain|week|wind|uv|monday..sunday]
// All modes respect the day offset. Single line output except 'week'.

function httpsGet(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { rejectUnauthorized: false }, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                try { resolve(JSON.parse(data)); }
                catch { reject(new Error('Failed to parse JSON')); }
            });
        }).on('error', reject);
    });
}

async function geocodeCity(city) {
    const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;
    const data = await httpsGet(url);
    if (!data.results || data.results.length === 0) throw new Error(`City not found: ${city}`);
    return data.results[0];
}

async function fetchWeather(lat, lon) {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max,uv_index_max&temperature_unit=celsius&wind_speed_unit=kmh&forecast_days=7`;
    return await httpsGet(url);
}

const WEATHER_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Foggy', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Drizzle', 55: 'Dense drizzle',
    61: 'Slight rain', 63: 'Rain', 65: 'Heavy rain',
    71: 'Slight snow', 73: 'Snow', 75: 'Heavy snow', 77: 'Snow grains',
    80: 'Light showers', 81: 'Showers', 82: 'Heavy showers',
    85: 'Light snow showers', 86: 'Heavy snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with hail', 99: 'Severe thunderstorm'
};

function getDayLabel(offset) {
    if (offset === 0) return 'Today';
    if (offset === 1) return 'Tomorrow';
    const d = new Date(); d.setDate(d.getDate() + offset);
    return ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()];
}

function uvLabel(uv) {
    if (uv <= 2) return 'Low';
    if (uv <= 5) return 'Moderate';
    if (uv <= 7) return 'High';
    if (uv <= 10) return 'Very high';
    return 'Extreme';
}

function dayHeader(day, d) {
    return `${getDayLabel(day)}: ${d.time[day]}`;
}

function showCurrent(loc, weather) {
    const c = weather.current;
    console.log(`Temperature:   ${c.temperature_2m}°C (feels like ${c.apparent_temperature}°C)`);
    console.log(`Condition:     ${WEATHER_CODES[c.weather_code] || 'Unknown'}`);
    console.log(`Humidity:      ${c.relative_humidity_2m}%`);
    console.log(`Wind:          ${c.wind_speed_10m} km/h (${c.wind_direction_10m}°)`);
    console.log(`Precipitation: ${c.precipitation} mm`);
}

function showDay(loc, weather, day) {
    const d = weather.daily;
    if (day >= d.time.length) { console.log(`Max ${d.time.length - 1} days ahead.`); return; }
    console.log(dayHeader(day, d));
    console.log(`High:          ${d.temperature_2m_max[day]}°C`);
    console.log(`Low:           ${d.temperature_2m_min[day]}°C`);
    console.log(`Condition:     ${WEATHER_CODES[d.weather_code[day]] || 'Unknown'}`);
    console.log(`Wind:          up to ${d.wind_speed_10m_max[day]} km/h`);
    console.log(`Precipitation: ${d.precipitation_sum[day]} mm`);
}

function showRain(loc, weather, day) {
    const d = weather.daily;
    if (day >= d.time.length) { console.log(`Max ${d.time.length - 1} days ahead.`); return; }
    const rain = d.precipitation_sum[day];
    console.log(dayHeader(day, d));
    console.log(`Rain:          ${rain > 0 ? 'Yes, ' + rain + ' mm' : 'No'}`);
    console.log(`Condition:     ${WEATHER_CODES[d.weather_code[day]] || 'Unknown'}`);
}

function showWind(loc, weather, day) {
    const d = weather.daily;
    if (day >= d.time.length) { console.log(`Max ${d.time.length - 1} days ahead.`); return; }
    console.log(dayHeader(day, d));
    console.log(`Wind:          up to ${d.wind_speed_10m_max[day]} km/h`);
}

function showUV(loc, weather, day) {
    const d = weather.daily;
    if (day >= d.time.length) { console.log(`Max ${d.time.length - 1} days ahead.`); return; }
    const uv = d.uv_index_max[day];
    const hi = d.temperature_2m_max[day];
    const rain = d.precipitation_sum[day];
    const noRain = rain === 0;
    const warm = hi >= 22;
    let verdict = noRain && warm ? 'Beach OK' : (!noRain ? 'Rain expected' : 'Too cool');
    if (noRain && warm && uv >= 6) verdict += ', wear sunscreen';
    console.log(dayHeader(day, d));
    console.log(`UV Index:      ${uv} (${uvLabel(uv)})`);
    console.log(`Temperature:   ${hi}°C`);
    console.log(`Precipitation: ${rain} mm`);
    console.log(`Verdict:       ${verdict}`);
}

function showWeek(loc, weather) {
    const d = weather.daily;
    const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    for (let i = 0; i < d.time.length; i++) {
        const date = new Date(d.time[i]);
        const day = days[date.getUTCDay()];
        const dd = String(date.getUTCDate()).padStart(2, ' ');
        const cond = WEATHER_CODES[d.weather_code[i]] || 'Unknown';
        const rain = d.precipitation_sum[i];
        console.log(`${day} ${dd}: ${d.temperature_2m_max[i]}°C/${d.temperature_2m_min[i]}°C ${cond}${rain > 0 ? ', ' + rain + 'mm' : ''}`);
    }
}

async function getWeather(city, query, dayOffset) {
    try {
        const loc = await geocodeCity(city);
        const weather = await fetchWeather(loc.latitude, loc.longitude);

        console.log('<res>');

        if (query === 'rain')       showRain(loc, weather, dayOffset);
        else if (query === 'wind')  showWind(loc, weather, dayOffset);
        else if (query === 'uv')    showUV(loc, weather, dayOffset);
        else if (query === 'week')  showWeek(loc, weather);
        else if (dayOffset > 0)     showDay(loc, weather, dayOffset);
        else                        showCurrent(loc, weather);

        console.log('</res>');
    } catch (err) {
        console.error('Error:', err.message);
    }
}

// Convert day name to offset from today (nearest future occurrence)
function dayNameToOffset(name) {
    const days = { sun: 0, sunday: 0, mon: 1, monday: 1, tue: 2, tuesday: 2, wed: 3, wednesday: 3, thu: 4, thursday: 4, fri: 5, friday: 5, sat: 6, saturday: 6 };
    const target = days[name.toLowerCase()];
    if (target === undefined) return null;
    const today = new Date().getDay();
    let offset = target - today;
    if (offset <= 0) offset += 7;
    return offset;
}

// Parse args: node weather_api.js [city] [query] [+day|dayname]
let city = 'London';
let query = null;
let dayOffset = 0;
const MODES = ['rain', 'week', 'wind', 'uv'];

for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (arg.startsWith('+')) {
        const n = parseInt(arg.substring(1));
        if (!isNaN(n)) dayOffset = n;
    } else if (MODES.includes(arg.toLowerCase())) {
        query = arg.toLowerCase();
    } else if (dayNameToOffset(arg) !== null) {
        dayOffset = dayNameToOffset(arg);
    } else {
        city = arg;
    }
}

getWeather(city, query, dayOffset);
