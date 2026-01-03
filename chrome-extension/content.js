/**
 * Content Script - Extraction des commentaires YouTube
 * S'exécute sur les pages YouTube pour extraire les commentaires
 */

console.log('🎬 YouTube Sentiment Analyzer - Content Script chargé');

/**
 * Extrait tous les commentaires visibles sur la page YouTube
 */
function extractComments() {
  console.log('📥 Extraction des commentaires...');
  
  const comments = [];
  
  // Sélecteur pour les commentaires YouTube
  const commentElements = document.querySelectorAll('ytd-comment-thread-renderer');
  
  console.log(`✅ ${commentElements.length} commentaires trouvés`);
  
  commentElements.forEach((element, index) => {
    try {
      // Extraire le texte du commentaire
      const commentTextElement = element.querySelector('#content-text');
      
      if (commentTextElement) {
        const text = commentTextElement.innerText.trim();
        
        // Extraire les métadonnées supplémentaires
        const authorElement = element.querySelector('#author-text');
        const author = authorElement ? authorElement.innerText.trim() : 'Anonyme';
        
        const likesElement = element.querySelector('#vote-count-middle');
        const likes = likesElement ? likesElement.innerText.trim() : '0';
        
        const timeElement = element.querySelector('.published-time-text a');
        const time = timeElement ? timeElement.innerText.trim() : '';
        
        if (text && text.length > 0) {
          comments.push({
            id: `comment_${index}`,
            text: text,
            author: author,
            likes: likes,
            time: time
          });
        }
      }
    } catch (error) {
      console.error(`Erreur lors de l'extraction du commentaire ${index}:`, error);
    }
  });
  
  console.log(`✅ ${comments.length} commentaires extraits avec succès`);
  return comments;
}

/**
 * Fait défiler la page pour charger plus de commentaires
 */
async function scrollToLoadComments(maxScrolls = 3) {
  console.log('📜 Défilement pour charger plus de commentaires...');
  
  // Trouver la section des commentaires
  const commentsSection = document.querySelector('ytd-comments#comments');
  
  if (!commentsSection) {
    console.log('⚠️ Section de commentaires non trouvée');
    return;
  }
  
  for (let i = 0; i < maxScrolls; i++) {
    // Défiler jusqu'à la fin de la section
    commentsSection.scrollIntoView({ behavior: 'smooth', block: 'end' });
    
    // Attendre que les nouveaux commentaires se chargent
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    console.log(`✅ Défilement ${i + 1}/${maxScrolls} effectué`);
  }
}

/**
 * Obtient les informations de la vidéo
 */
function getVideoInfo() {
  const titleElement = document.querySelector('h1.ytd-watch-metadata yt-formatted-string');
  const channelElement = document.querySelector('ytd-channel-name#channel-name a');
  const viewsElement = document.querySelector('ytd-watch-info-text span.view-count');
  
  return {
    title: titleElement ? titleElement.innerText.trim() : 'Titre non disponible',
    channel: channelElement ? channelElement.innerText.trim() : 'Chaîne inconnue',
    views: viewsElement ? viewsElement.innerText.trim() : '0',
    url: window.location.href
  };
}

/**
 * Écoute les messages du popup
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('📨 Message reçu:', request);
  
  if (request.action === 'extractComments') {
    console.log('🎯 Action: Extraction des commentaires');
    
    // Extraire immédiatement les commentaires visibles
    const comments = extractComments();
    const videoInfo = getVideoInfo();
    
    // Envoyer la réponse
    sendResponse({
      success: true,
      comments: comments,
      videoInfo: videoInfo,
      count: comments.length
    });
    
  } else if (request.action === 'scrollAndExtract') {
    console.log('🎯 Action: Défilement et extraction');
    
    // Utiliser une fonction async
    (async () => {
      try {
        await scrollToLoadComments(3);
        const comments = extractComments();
        const videoInfo = getVideoInfo();
        
        sendResponse({
          success: true,
          comments: comments,
          videoInfo: videoInfo,
          count: comments.length
        });
      } catch (error) {
        console.error('❌ Erreur:', error);
        sendResponse({
          success: false,
          error: error.message
        });
      }
    })();
    
    // Retourner true pour indiquer une réponse asynchrone
    return true;
  }
});

console.log('✅ Content Script prêt à extraire les commentaires');