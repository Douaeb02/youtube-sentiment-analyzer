/**
 * Popup Script - Logique de l'interface utilisateur
 */

console.log('🎬 Popup Script chargé');

// État global
let currentComments = [];  
let currentPredictions = [];
let currentFilter = 'all';

// ============================================================================
// Initialisation
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  console.log('🔔 DOMContentLoaded - Initialisation du popup');

  try {
    // 🔥 Réveil forcé du Service Worker
    console.log('⏰ Réveil du Service Worker...');
    await sendMessageToBackground({ action: 'wakeUp' });
    console.log('✅ Service Worker réveillé');
  } catch (e) {
    console.warn('⚠️ Erreur réveil SW:', e.message);
  }

  await loadSettings();
  await checkApiStatus();
  setupEventListeners();

  console.log('✅ Popup prêt');
});

// ============================================================================
// Configuration des Event Listeners
// ============================================================================

function setupEventListeners() {
  console.log('🎛️ Configuration des event listeners');
  
  document.getElementById('analyzeBtn').addEventListener('click', analyzeComments);
  document.getElementById('loadMoreBtn').addEventListener('click', loadMoreComments);
  document.getElementById('darkModeToggle').addEventListener('click', toggleDarkMode);
  
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => filterComments(e.target.dataset.filter));
  });
  
  document.getElementById('copyBtn').addEventListener('click', copyResults);
  
  document.getElementById('settingsBtn').addEventListener('click', () => {
    document.getElementById('settingsModal').classList.remove('hidden');
  });
  
  document.getElementById('closeModal').addEventListener('click', () => {
    document.getElementById('settingsModal').classList.add('hidden');
  });
  
  document.getElementById('saveSettings').addEventListener('click', saveSettings);
  
  console.log('✅ Event listeners configurés');
}

// ============================================================================
// Fonctions principales
// ============================================================================

/**
 * Analyse les commentaires de la page YouTube
 */
async function analyzeComments() {
  console.log('🎯 ========== DÉBUT ANALYSE ==========');
  
  showLoading(true);
  showStatus('Extraction des commentaires...', 'info');
  
  try {
    // 1️⃣ Obtenir l'onglet actif
    console.log('📍 Étape 1: Récupération de l\'onglet actif');
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    console.log('✅ Onglet trouvé:', tab.url);
    
    // 2️⃣ Vérifier qu'on est sur YouTube
    if (!tab.url.includes('youtube.com/watch')) {
      console.error('❌ Pas sur une vidéo YouTube');
      showStatus('❌ Veuillez ouvrir une vidéo YouTube', 'error');
      showLoading(false);
      return;
    }
    
    // 3️⃣ Extraire les commentaires via le content script
    console.log('📍 Étape 2: Extraction des commentaires');
    showStatus('Extraction des commentaires...', 'info');
    
    let response;
    try {
      response = await chrome.tabs.sendMessage(tab.id, { action: 'extractComments' });
      console.log('✅ Réponse reçue du content script:', response);
    } catch (error) {
      console.error('❌ Erreur sendMessage au content script:', error);
      showStatus('❌ Erreur: Le content script n\'a pas répondu. Rechargez la page YouTube.', 'error');
      showLoading(false);
      return;
    }
    
    if (!response || !response.success) {
      console.error('❌ Échec extraction:', response);
      showStatus('❌ Erreur lors de l\'extraction', 'error');
      showLoading(false);
      return;
    }
    
    currentComments = response.comments;
    console.log(`✅ ${currentComments.length} commentaires extraits`);
    
    // 4️⃣ Afficher les infos de la vidéo
    if (response.videoInfo) {
      displayVideoInfo(response.videoInfo);
    }
    
    if (currentComments.length === 0) {
      showStatus('⚠️ Aucun commentaire trouvé. Essayez "Charger plus"', 'error');
      showLoading(false);
      return;
    }
    
    // 5️⃣ Préparer les textes pour l'analyse
    console.log('📍 Étape 3: Préparation pour l\'analyse');
    showStatus(`Analyse de ${currentComments.length} commentaires...`, 'info');
    const texts = currentComments.map(c => c.text);
    console.log('✅ Textes préparés:', texts.length);
    
    // 6️⃣ Réveiller le Service Worker
    console.log('📍 Étape 4: Réveil du Service Worker');
    try {
      await sendMessageToBackground({ action: 'wakeUp' });
      console.log('✅ Service Worker réveillé');
    } catch (e) {
      console.warn('⚠️ Réveil SW échoué:', e);
    }
    
    // 7️⃣ Vérifier la santé de l'API
    console.log('📍 Étape 5: Vérification de l\'API');
    showStatus('Vérification de l\'API...', 'info');
    
    let health;
    try {
      health = await sendMessageToBackground({ action: 'checkHealth' });
      console.log('✅ Réponse health check:', health);
    } catch (error) {
      console.error('❌ Erreur health check:', error);
      showStatus('❌ Impossible de contacter l\'API. Vérifiez qu\'elle est démarrée.', 'error');
      showLoading(false);
      return;
    }
    
    if (!health.success || !health.healthy) {
      console.error('❌ API non disponible:', health);
      showStatus('❌ API indisponible. Lancez: python -m src.api.run_api', 'error');
      showLoading(false);
      return;
    }
    
    console.log('✅ API opérationnelle');
    
    // 8️⃣ Envoyer pour analyse
    console.log('📍 Étape 6: Envoi à l\'API pour analyse');
    showStatus(`Analyse en cours (${texts.length} commentaires)...`, 'info');
    
    let analysisResult;
    try {
      analysisResult = await sendMessageToBackground({
        action: 'analyzeSentiment',
        comments: texts
      });
      console.log('✅ Réponse analyse:', analysisResult);
    } catch (error) {
      console.error('❌ Erreur analyse:', error);
      showStatus('❌ Erreur lors de l\'analyse: ' + error.message, 'error');
      showLoading(false);
      return;
    }
    
    if (!analysisResult.success) {
      console.error('❌ Analyse échouée:', analysisResult);
      showStatus('❌ Erreur: ' + (analysisResult.error || 'Analyse échouée'), 'error');
      showLoading(false);
      return;
    }
    
    // 9️⃣ Combiner les résultats
    console.log('📍 Étape 7: Traitement des résultats');
    currentPredictions = analysisResult.data.predictions.map((pred, index) => ({
      ...currentComments[index],
      ...pred
    }));
    
    console.log('✅ Résultats combinés:', currentPredictions.length);
    
    // 🔟 Afficher les résultats
    console.log('📍 Étape 8: Affichage des résultats');
    displayStatistics(analysisResult.data.statistics);
    displayComments(currentPredictions);
    
    showStatus(`✅ ${currentPredictions.length} commentaires analysés!`, 'success');
    showLoading(false);
    
    console.log('🎯 ========== FIN ANALYSE ==========');
    
  } catch (error) {
    console.error('❌ ERREUR CRITIQUE:', error);
    console.error('Stack:', error.stack);
    showStatus('❌ Erreur: ' + error.message, 'error');
    showLoading(false);
  }
}

/**
 * Charge plus de commentaires en défilant la page
 */
async function loadMoreComments() {
  console.log('📜 Chargement de plus de commentaires');
  showLoading(true);
  showStatus('Chargement de plus de commentaires...', 'info');
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const response = await chrome.tabs.sendMessage(tab.id, { 
      action: 'scrollAndExtract' 
    });
    
    if (response.success) {
      showStatus(`✅ ${response.count} commentaires chargés. Cliquez sur "Analyser".`, 'success');
    } else {
      showStatus('❌ Erreur lors du chargement', 'error');
    }
    
  } catch (error) {
    console.error('❌ Erreur:', error);
    showStatus('❌ Erreur: ' + error.message, 'error');
  } finally {
    showLoading(false);
  }
}

// ============================================================================
// Affichage des résultats
// ============================================================================

function displayVideoInfo(videoInfo) {
  const container = document.getElementById('videoInfo');
  document.getElementById('videoTitle').textContent = videoInfo.title;
  document.getElementById('videoChannel').textContent = `📺 ${videoInfo.channel}`;
  container.classList.remove('hidden');
}

function displayStatistics(stats) {
  document.getElementById('positiveCount').textContent = stats.positive;
  document.getElementById('neutralCount').textContent = stats.neutral;
  document.getElementById('negativeCount').textContent = stats.negative;
  
  document.getElementById('positivePercent').textContent = stats.positive_percent + '%';
  document.getElementById('neutralPercent').textContent = stats.neutral_percent + '%';
  document.getElementById('negativePercent').textContent = stats.negative_percent + '%';
  
  document.getElementById('positiveBar').style.width = stats.positive_percent + '%';
  document.getElementById('neutralBar').style.width = stats.neutral_percent + '%';
  document.getElementById('negativeBar').style.width = stats.negative_percent + '%';
  
  document.getElementById('positiveBarPercent').textContent = stats.positive_percent + '%';
  document.getElementById('neutralBarPercent').textContent = stats.neutral_percent + '%';
  document.getElementById('negativeBarPercent').textContent = stats.negative_percent + '%';
  
  document.getElementById('totalComments').textContent = stats.total;
  document.getElementById('avgConfidence').textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
  
  document.getElementById('statistics').classList.remove('hidden');
  document.getElementById('filters').classList.remove('hidden');
}

function displayComments(predictions) {
  const container = document.getElementById('commentsList');
  container.innerHTML = '';
  
  let filtered = predictions;
  if (currentFilter !== 'all') {
    filtered = predictions.filter(p => {
      if (currentFilter === 'positive') return p.sentiment === 'Positif';
      if (currentFilter === 'neutral') return p.sentiment === 'Neutre';
      if (currentFilter === 'negative') return p.sentiment === 'Négatif';
      return true;
    });
  }
  
  filtered.forEach(comment => {
    const item = createCommentElement(comment);
    container.appendChild(item);
  });
  
  document.getElementById('commentsContainer').classList.remove('hidden');
}

function createCommentElement(comment) {
  const div = document.createElement('div');
  div.className = 'comment-item';
  
  const emoji = {
    'Positif': '😊',
    'Neutre': '😐',
    'Négatif': '😞'
  }[comment.sentiment];
  
  div.innerHTML = `
    <div class="comment-header">
      <div class="comment-sentiment">
        <span>${emoji}</span>
        <span>${comment.sentiment}</span>
      </div>
      <div class="comment-confidence">
        ${(comment.confidence * 100).toFixed(1)}%
      </div>
    </div>
    <div class="comment-text">${escapeHtml(comment.text)}</div>
    <div class="comment-meta">
      <span>👤 ${escapeHtml(comment.author)}</span>
      <span>👍 ${comment.likes}</span>
      <span>🕐 ${comment.time}</span>
    </div>
  `;
  
  return div;
}

function filterComments(filter) {
  console.log('🔍 Filtre:', filter);
  currentFilter = filter;
  
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filter);
  });
  
  displayComments(currentPredictions);
}

async function copyResults() {
  let text = '📊 ANALYSE DE SENTIMENT YOUTUBE\n';
  text += '='.repeat(50) + '\n\n';
  
  const stats = {
    positif: parseInt(document.getElementById('positiveCount').textContent),
    neutre: parseInt(document.getElementById('neutralCount').textContent),
    negatif: parseInt(document.getElementById('negativeCount').textContent)
  };
  
  text += `Positif: ${stats.positif}\n`;
  text += `Neutre: ${stats.neutre}\n`;
  text += `Négatif: ${stats.negatif}\n\n`;
  
  text += 'COMMENTAIRES:\n';
  text += '-'.repeat(50) + '\n\n';
  
  currentPredictions.forEach((comment, i) => {
    const emoji = {
      'Positif': '😊',
      'Neutre': '😐',
      'Négatif': '😞'
    }[comment.sentiment];
    
    text += `${i + 1}. ${emoji} ${comment.sentiment} (${(comment.confidence * 100).toFixed(1)}%)\n`;
    text += `   "${comment.text}"\n`;
    text += `   Par: ${comment.author}\n\n`;
  });
  
  try {
    await navigator.clipboard.writeText(text);
    showStatus('✅ Résultats copiés!', 'success');
    setTimeout(() => showStatus('', 'info'), 2000);
  } catch (error) {
    showStatus('❌ Erreur lors de la copie', 'error');
  }
}

// ============================================================================
// Paramètres
// ============================================================================

async function loadSettings() {
  const settings = await chrome.storage.local.get(['darkMode', 'apiType']);
  
  if (settings.darkMode) {
    document.body.classList.add('dark-mode');
    document.getElementById('darkModeToggle').textContent = '☀️';
  }
  
  if (settings.apiType) {
    const radio = document.querySelector(`input[value="${settings.apiType}"]`);
    if (radio) radio.checked = true;
  }
}

async function saveSettings() {
  const apiType = document.querySelector('input[name="apiType"]:checked').value;
  
  await chrome.storage.local.set({ apiType });
  
  await sendMessageToBackground({
    action: 'switchApi',
    apiType: apiType
  });
  
  showStatus('✅ Paramètres sauvegardés', 'success');
  document.getElementById('settingsModal').classList.add('hidden');
  
  await checkApiStatus();
}

async function toggleDarkMode() {
  const isDark = document.body.classList.toggle('dark-mode');
  document.getElementById('darkModeToggle').textContent = isDark ? '☀️' : '🌙';
  
  await chrome.storage.local.set({ darkMode: isDark });
}

async function checkApiStatus() {
  console.log('🏥 Vérification du statut de l\'API');
  const statusElement = document.getElementById('apiStatus');
  
  try {
    const result = await sendMessageToBackground({ action: 'checkHealth' });
    console.log('✅ Résultat health check:', result);
    
    if (result.success && result.healthy) {
      statusElement.textContent = '🟢 API connectée';
      statusElement.style.color = '#48bb78';
    } else {
      statusElement.textContent = '🔴 API déconnectée';
      statusElement.style.color = '#f56565';
    }
  } catch (error) {
    console.error('❌ Erreur health check:', error);
    statusElement.textContent = '🔴 API déconnectée';
    statusElement.style.color = '#f56565';
  }
}

// ============================================================================
// Utilitaires
// ============================================================================

function showLoading(show) {
  document.getElementById('loadingSpinner').classList.toggle('hidden', !show);
  document.getElementById('analyzeBtn').disabled = show;
  document.getElementById('loadMoreBtn').disabled = show;
}

function showStatus(message, type = 'info') {
  const statusBar = document.getElementById('statusBar');
  const statusText = document.getElementById('statusText');
  
  statusText.textContent = message;
  statusBar.className = `status-bar ${type}`;
  statusBar.classList.toggle('hidden', !message);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function sendMessageToBackground(message) {
  console.log('📤 Envoi message au background:', message.action);
  
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        console.error('❌ Erreur runtime:', chrome.runtime.lastError.message);
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        console.log('📥 Réponse reçue:', response);
        resolve(response);
      }
    });
  });
}

console.log('✅ Popup Script prêt');