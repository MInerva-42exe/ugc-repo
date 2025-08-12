document.addEventListener('DOMContentLoaded', () => {
  const chatForm   = document.getElementById('chat-form');
  const chatInput  = document.getElementById('chat-input');
  const chatWindow = document.getElementById('chat-window');

  const API_URL    = '/api/query';
  const HEALTH_URL = '/api/health';

  function addMessage(text, sender) {
    const m = document.createElement('div');
    m.className = `message ${sender}-message`;
    m.textContent = text;
    chatWindow.appendChild(m);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // Small helpers
  const make = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text) el.textContent = text;
    return el;
  };

  const addPills = (parent, items, containerCls, pillCls) => {
    if (!items || !items.length) return;
    const wrap = make('div', containerCls);
    items.filter(Boolean).forEach(txt => wrap.appendChild(make('span', pillCls, txt)));
    parent.appendChild(wrap);
  };

  // Story card (two-column)
  function renderStoryCard(story) {
    const {
      organization_name, org_logo, org_photo,
      customer_name, customer_designation, customer_linkedin,
      case_study_link,
      is_enterprise_case_study, is_reference_customer,
      summary, problem, solution, outcome, raw_testimonial_quote,
      country, industry_tags, product
    } = story || {};

    const card = make('div', 'story');

    // ----- Left column -----
    const left = make('div', 'story-left');

    // Logo
    const logoWrap = make('div', 'story-logo');
    if (org_logo) {
      const img = document.createElement('img');
      img.src = org_logo;
      img.alt = organization_name ? `${organization_name} logo` : 'Organization logo';
      logoWrap.appendChild(img);
    }
    left.appendChild(logoWrap);

    // Solution
    if (solution || product) {
      const sol = make('div', 'story-solution');
      sol.textContent = `Solution: ${solution || product}`;
      left.appendChild(sol);
    }

    // Badges (Enterprise/Reference)
    const badges = [];
    if (is_enterprise_case_study) badges.push('Enterprise');
    if (is_reference_customer)    badges.push('Reference Customer');
    addPills(left, badges, 'badges', 'badge');

    // Country + Industry bubbles — FIXED
    const metaChips = [];
    if (Array.isArray(country) && country.length) metaChips.push(...country);
    if (Array.isArray(industry_tags) && industry_tags.length) metaChips.push(...industry_tags);
    addPills(left, metaChips, 'tags-container', 'tag-bubble');

    // ----- Right column -----
    const right = make('div', 'story-right');

    // Title (org)
    if (organization_name) {
      const title = make('a', 'story-title', organization_name);
      if (case_study_link) { title.href = case_study_link; title.target = '_blank'; title.rel = 'noopener'; }
      else { title.href = '#'; title.onclick = e => e.preventDefault(); }
      right.appendChild(title);
    }

    // Person row
    if (customer_name || customer_designation || customer_linkedin || org_photo) {
      const person = make('div', 'person');

      if (org_photo) {
        const av = make('div', 'person-avatar');
        const img = document.createElement('img');
        img.src = org_photo;
        img.alt = customer_name ? `${customer_name} photo` : 'Customer photo';
        av.appendChild(img);
        person.appendChild(av);
      }

      const info = make('div', 'person-block');
      if (customer_name) info.appendChild(make('div', 'person-name', customer_name));

      const metaRow = make('div', 'person-meta');
      if (customer_designation) metaRow.appendChild(make('span', null, customer_designation));
      if (customer_linkedin) {
        const link = make('span', 'person-link');
        const a = document.createElement('a');
        a.href = customer_linkedin; a.textContent = 'LinkedIn'; a.target = '_blank'; a.rel = 'noopener';
        if (metaRow.textContent) link.style.marginLeft = '8px';
        link.appendChild(a);
        metaRow.appendChild(link);
      }
      if (metaRow.textContent || metaRow.querySelector('a')) info.appendChild(metaRow);

      person.appendChild(info);
      right.appendChild(person);
    }

    // Details
    const details = make('div', 'details-grid');
    const addDetail = (label, value) => {
      if (!value) return;
      const block = make('div', 'detail-block');
      const lab = make('div', 'detail-label', label);
      const val = make('div', null, value);
      block.appendChild(lab); block.appendChild(val);
      details.appendChild(block);
    };
    addDetail('Summary', summary);
    addDetail('Challenge', problem);
    addDetail('Outcome', outcome);
    if (details.children.length) right.appendChild(details);

    if (raw_testimonial_quote) {
      right.appendChild(make('div', 'quote', `"${raw_testimonial_quote}"`));
    }

    // CTA row
    if (case_study_link) {
      const actions = make('div', 'card-actions');
      const btn = make('a', 'btn', 'Read full case study');
      btn.href = case_study_link; btn.target = '_blank'; btn.rel = 'noopener';
      actions.appendChild(btn);
      right.appendChild(actions);
    }

    // assemble
    card.appendChild(left);
    card.appendChild(right);
    return card;
  }

  function displayStories(stories) {
    if (!stories || stories.length === 0) return;

    const wrap = document.createElement('div');
    wrap.className = 'stories';

    const head = document.createElement('div');
    head.className = 'stories-header';
    head.textContent = `Found ${stories.length} relevant customer stories:`;
    wrap.appendChild(head);

    stories.forEach(story => wrap.appendChild(renderStoryCard(story)));

    chatWindow.appendChild(wrap);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function testBackendConnection() {
    try { const r = await fetch(HEALTH_URL); return r.ok; }
    catch { return false; }
  }

  // Interaction
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userQuery = chatInput.value.trim();
    if (!userQuery) return;

    addMessage(userQuery, 'user');
    chatInput.value = ''; chatInput.disabled = true;

    const loading = document.createElement('div');
    loading.className = 'message bot-message';
    loading.textContent = 'Processing…';
    chatWindow.appendChild(loading);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, history: [] }),
      });

      chatWindow.removeChild(loading);
      if (!res.ok) throw new Error(`Server responded with status ${res.status}`);

      const data = await res.json();
      const hasStories = data.found_stories && data.found_stories.length > 0;

      if (hasStories) displayStories(data.found_stories);
      else addMessage(data.message, 'bot');
    } catch (err) {
      chatWindow.removeChild(loading);
      addMessage('Sorry, something went wrong. Please try again.', 'bot');
      console.error(err);
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });

  // Boot
  (async function init(){
    addMessage("Hello! I'm your Customer Success Stories assistant. How can I help?", 'bot');
    const ok = await testBackendConnection();
    if (!ok) addMessage("Warning: I can't reach the backend right now.", 'bot');
    chatInput.focus();
  })();

  // Enter submits
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });
});
