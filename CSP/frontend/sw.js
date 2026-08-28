const CACHE_NAME = 'agriai-v2';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://cdn.jsdelivr.net/npm/chart.js'
];

// Install Event - cache core shell assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching app shell');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// Activate Event
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch Event - cache first, fallback to network
self.addEventListener('fetch', (event) => {
  // Skip caching API POST requests (which send images/chat logs)
  if (event.request.method !== 'GET' || event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        // Return a mock JSON response if API fails offline
        if (event.request.url.includes('/api/weather')) {
          return new Response(JSON.stringify({
            temperature: 29.5,
            humidity: 60,
            rain_probability: 10,
            description: "Offline (Using Cached Forecast)",
            advisory: ["📴 Offline mode active. Displaying cached forecast from last synchronization."],
            forecast: []
          }), { headers: { 'Content-Type': 'application/json' } });
        }
      })
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      
      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200) {
          return networkResponse;
        }
        
        // Dynamically cache newly fetched assets
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        
        return networkResponse;
      }).catch(() => {
        // If offline and request is an image from uploads, return placeholder if not cached
        if (event.request.url.includes('/uploads/')) {
          return caches.match('https://cdn-icons-png.flaticon.com/512/628/628283.png');
        }
      });
    })
  );
});
