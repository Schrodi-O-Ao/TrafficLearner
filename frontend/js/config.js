// GitHub Pages serves only files, so point those pages to the Render API.
// Local and Render-hosted pages use their own origin unless overridden.
if (!window.TRAFFIC_API_BASE && window.location.hostname.endsWith('.github.io')) {
    // If Render gives the service a different hostname, change this URL.
    window.TRAFFIC_API_BASE = 'https://traffic-learner-api.onrender.com';
}
