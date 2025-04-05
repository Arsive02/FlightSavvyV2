import { useEffect } from 'react';

const WatsonAssistantScript = () => {
  useEffect(() => {
    const script = document.createElement('script');
    script.innerHTML = `
  window.watsonAssistantChatOptions = {
    integrationID: "c28a7c8b-8973-4491-9282-2d8ea6e2e724", // The ID of this integration.
    region: "au-syd", // The region your integration is hosted in.
    serviceInstanceID: "429ed1c0-3386-40ce-b50c-8b6eae6d214a", // The ID of your service instance.
    onLoad: async (instance) => { await instance.render(); }
  };
  setTimeout(function(){
    const t=document.createElement('script');
    t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
  });
    `;
    script.async = true;
    
    document.head.appendChild(script);
    
    return () => {
      document.head.removeChild(script);
    };
  }, []);

  return null;
};

export default WatsonAssistantScript;