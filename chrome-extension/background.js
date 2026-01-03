/**
 * Background Script - Service Worker
 * Gère les tâches en arrière-plan de l'extension
 */

console.log('🚀 YouTube Sentiment Analyzer - Background Script chargé');

// Configuration de l'API
const API_CONFIG = {
  local: 'http://localhost:8000',
  production: 'https://douae8bz-youtube-sentiment-analyzer.hf.space',
  current: 'local'
};

// 🔥 KEEPALIVE: Empêche le Service Worker de s'endormir
let keepAliveInterval = null;

function startKeepAlive() {
  if (keepAliveInterval) return;
  
  keepAliveInterval = setInterval(() => {
    console.log('💓 Keepalive ping');
  }, 20000); // Toutes les 20 secondes
}

function stopKeepAlive() {
  if (keepAliveInterval) {
    clearInterval(keepAliveInterval);
    keepAliveInterval = null;
  }
}

// Démarrer le keepalive au chargement
startKeepAlive();

/**
 * Obtient l'URL de l'API actuelle
 */
function getApiUrl() {
  return API_CONFIG[API_CONFIG.current];
}

/**
 * Envoie les commentaires à l'API pour analyse
 */
async function analyzeSentiment(comments) {
  const apiUrl = getApiUrl();
  const endpoint = `${apiUrl}/predict_batch`;
  
  console.log(`📡 Envoi de ${comments.length} commentaires à ${endpoint}`);
  
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        comments: comments
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ Analyse terminée:', data);
    
    return {
      success: true,
      data: data
    };
    
  } catch (error) {
    console.error('❌ Erreur lors de l\'analyse:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Vérifie l'état de l'API
 */
async function checkApiHealth() {
  const apiUrl = getApiUrl();
  const endpoint = `${apiUrl}/health`;
  
  console.log(`🏥 Health check: ${endpoint}`);
  
  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ Health check OK:', data);
    
    return {
      success: true,
      healthy: data.status === 'healthy',
      data: data
    };
  } catch (error) {
    console.error('❌ Erreur health check:', error);
    return {
      success: false,
      healthy: false,
      error: error.message
    };
  }
}

/**
 * Écoute les messages des autres parties de l'extension
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('📨 Message reçu dans background:', request.action);
  
  // 🔥 IMPORTANT: Garder le Service Worker éveillé pendant l'opération
  startKeepAlive();
  
  if (request.action === 'wakeUp') {
    console.log('☀️ Service Worker réveillé');
    sendResponse({ success: true, message: 'Service Worker actif' });
    return true;
  }
  
  else if (request.action === 'analyzeSentiment') {
    console.log(`🎯 Analyse de ${request.comments?.length || 0} commentaires`);
    analyzeSentiment(request.comments)
      .then(result => {
        console.log('📤 Envoi de la réponse d\'analyse');
        sendResponse(result);
      })
      .catch(error => {
        console.error('❌ Erreur analyse:', error);
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Indique une réponse asynchrone
    
  } else if (request.action === 'checkHealth') {
    console.log('🎯 Vérification de la santé de l\'API');
    checkApiHealth()
      .then(result => {
        console.log('📤 Envoi de la réponse health check');
        sendResponse(result);
      })
      .catch(error => {
        console.error('❌ Erreur health check:', error);
        sendResponse({ success: false, healthy: false, error: error.message });
      });
    
    return true;
    
  } else if (request.action === 'switchApi') {
    API_CONFIG.current = request.apiType;
    console.log(`🔄 API changée vers: ${API_CONFIG.current} (${getApiUrl()})`);
    sendResponse({ success: true, apiUrl: getApiUrl() });
    return true;
  }
  
  // Action inconnue
  console.warn('⚠️ Action inconnue:', request.action);
  sendResponse({ success: false, error: 'Action inconnue' });
  return true;
});

/**
 * Initialisation au démarrage
 */
chrome.runtime.onInstalled.addListener(() => {
  console.log('🎉 Extension installée avec succès!');
  
  chrome.storage.local.set({
    apiType: 'local',
    darkMode: false,
    autoAnalyze: false
  });
  
  startKeepAlive();
});

/**
 * Au démarrage de Chrome
 */
chrome.runtime.onStartup.addListener(() => {
  console.log('🌅 Chrome démarré - Réactivation du Service Worker');
  startKeepAlive();
});

/**
 * Nettoyage à la suspension
 */
self.addEventListener('suspend', () => {
  console.log('💤 Service Worker en suspension');
  stopKeepAlive();
});

console.log('✅ Background Script prêt et actif');