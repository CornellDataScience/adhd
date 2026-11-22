const form = document.querySelector('#chat-form');
const sendIcon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 20V4M5 11l7-7 7 7" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
form.querySelector('button').innerHTML = sendIcon;
form.querySelector('button').setAttribute('aria-label', 'Send message');
const prompt = document.querySelector('#prompt');
const conversation = document.querySelector('#conversation');
const demoStyles = document.createElement('link');
demoStyles.rel = 'stylesheet';
demoStyles.href = './demo.css?v=20261122-14';
document.head.append(demoStyles);
// Embed the live project archive so the demo keeps the actual CDS project page.
const projectsFrame = document.createElement('iframe');
projectsFrame.src = 'https://cornelldata.science/projects';
projectsFrame.title = 'Cornell Data Science projects';
projectsFrame.className = 'cds-project-page';
document.body.prepend(projectsFrame);
const launcher = document.createElement('button');
launcher.className = 'chat-launcher';
launcher.textContent = 'Ask CDS';
launcher.setAttribute('aria-expanded', 'false');
document.body.append(launcher);
const assistant = document.querySelector('#assistant');
document.body.append(assistant);
assistant.classList.add('demo-chat');
const closeChat = document.createElement('button');
closeChat.className = 'close-chat';
closeChat.textContent = '×';
closeChat.setAttribute('aria-label', 'Close chatbot');
assistant.prepend(closeChat);
function toggleChat(open) {
  assistant.classList.toggle('chat-open', open);
  launcher.setAttribute('aria-expanded', String(open));
  if (open) prompt.focus();
}
launcher.addEventListener('click', () => toggleChat(!assistant.classList.contains('chat-open')));
closeChat.addEventListener('click', () => toggleChat(false));
assistant.querySelector('.panel-note').textContent = 'Demo: responses use the model endpoint. Repository retrieval is not connected yet.';
assistant.querySelector('.mono').textContent = 'MODEL DEMO';
assistant.querySelector('.assistant-heading').hidden = true;
assistant.querySelector('.panel-top').hidden = true;
assistant.querySelector('.chips').hidden = true;
assistant.querySelector('.panel-note').hidden = true;
conversation.replaceChildren();
const chatTitle = document.createElement('div');
chatTitle.className = 'simple-chat-title';
chatTitle.textContent = 'Ask CDS';
assistant.insertBefore(chatTitle, closeChat);
document.querySelector('#menu-toggle').addEventListener('click', () => document.querySelector('#menu').classList.toggle('open'));
document.querySelector('#search-toggle').addEventListener('click', () => document.querySelector('#prompt').focus());

function addMessage(text, role) {
  const item = document.createElement('div');
  item.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  item.innerHTML = role === 'user'
    ? `<div><div class="message-meta">YOU <span>NOW</span></div><p>${text.replace(/</g, '&lt;')}</p></div>`
    : `<div class="avatar">C</div><div><div class="message-meta">CDS ASSISTANT <span>NOW</span></div><p>${text.replace(/</g, '&lt;')}</p><div class="citation">↳ Connect /generate to show grounded answers</div></div>`;
  conversation.appendChild(item);
  conversation.scrollTop = conversation.scrollHeight;
}

async function ask(question) {
  const send = form.querySelector('button');
  if (send.disabled) return;
  send.disabled = true;
  send.innerHTML = '<svg class="loading-square" width="9" height="9" viewBox="0 0 12 12" aria-hidden="true"><rect width="12" height="12" rx="1" fill="currentColor"/></svg>';
  send.setAttribute('aria-label', 'Generating response');
  addMessage(question, 'user');
  const thinking = document.createElement('div');
  thinking.className = 'thinking';
  thinking.setAttribute('role', 'status');
  thinking.setAttribute('aria-label', 'Generating response');
  thinking.innerHTML = '<span></span><span></span><span></span>';
  conversation.append(thinking);
  conversation.scrollTop = conversation.scrollHeight;
  prompt.value = '';
  try {
    const response = await fetch('http://127.0.0.1:8000/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: question, max_new_tokens: 96})
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Model request failed');
    }
    const data = await response.json();
    thinking.remove();
    addMessage(data.answer || 'The model returned an empty answer.', 'assistant');
  } catch (error) {
    thinking.remove();
    addMessage(error instanceof TypeError ? 'Cannot reach the model server. Please check that it is running on port 8000.' : error.message, 'assistant');
  } finally {
    thinking.remove();
    send.disabled = false;
    send.innerHTML = sendIcon;
    send.setAttribute('aria-label', 'Send message');
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); const value = prompt.value.trim(); if (value) ask(value); });
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => ask(button.dataset.question)));
prompt.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
