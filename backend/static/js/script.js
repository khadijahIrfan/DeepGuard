/* ================================================================
   DEEPGUARD — SHARED APPLICATION JAVASCRIPT (script.js)
   ================================================================ */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initSidebarToggle();
  initHeaderPhrases();
  initNotificationSystem();
  initAnalyzeForm();
  initRegisterFormValidation();
  initHistoryFilters();
});

/* ================================================================
   DYNAMIC HEADER PHRASES ROTATION TICKER
   ================================================================ */
function initHeaderPhrases() {
  const elem = document.getElementById('headerDynamicPhrase');
  if (!elem) return;

  const phrases = [
    "🛡️ Shielding Truth with AI • Multimodal Intelligence",
    "🔍 DistilBERT Transformer + MobileNetV2 Vision Engine",
    "⚡ Real-Time Cross-Modal Context & EXIF Forensic Scanner",
    "💡 Explainable AI (XAI) for Digital Media Authenticity",
    "🌐 Defense Against Fake News, DeepFakes & WhatsApp Rumors"
  ];

  let idx = 0;
  setInterval(() => {
    idx = (idx + 1) % phrases.length;
    elem.style.transition = 'opacity 0.25s ease';
    elem.style.opacity = '0';
    setTimeout(() => {
      elem.textContent = phrases[idx];
      elem.style.opacity = '1';
    }, 250);
  }, 5500);
}

/* ================================================================
   0. THEME TOGGLE CONTROLLER (Light / Dark)
   ================================================================ */
function initThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (!toggleBtn) return;

  // Read current theme from attribute or localStorage
  const currentTheme = document.documentElement.getAttribute('data-theme') || 
                       localStorage.getItem('deepguard_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);

  toggleBtn.addEventListener('click', () => {
    const active = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = active === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('deepguard_theme', nextTheme);

    // Sync to user profile in database
    fetch('/api/user/theme', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ theme: nextTheme })
    }).catch(err => console.log('Theme sync notice:', err));
  });
}

/* ================================================================
   1. SIDEBAR TOGGLE & HOVER SLIDE-OUT CONTROLLER
   ================================================================ */
function initSidebarToggle() {
  const toggleBtn = document.getElementById('sidebarToggleBtn');
  const pinBtn = document.getElementById('sidebarPinBtn');
  const mobileToggleBtn = document.getElementById('mobileToggleBtn');
  const sidebar = document.getElementById('sidebar');

  // Check stored preference in localStorage
  const savedState = localStorage.getItem('deepguard_sidebar_mode');

  // Default mode is collapsed (hover slide-out mode), unless user explicitly pinned it
  if (savedState === 'pinned') {
    document.body.classList.remove('sidebar-collapsed');
  } else {
    document.body.classList.add('sidebar-collapsed');
  }

  function toggleSidebarMode(e) {
    if (e) e.stopPropagation();
    document.body.classList.toggle('sidebar-collapsed');
    const isCollapsed = document.body.classList.contains('sidebar-collapsed');
    localStorage.setItem('deepguard_sidebar_mode', isCollapsed ? 'collapsed' : 'pinned');
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', toggleSidebarMode);
  }

  if (pinBtn) {
    pinBtn.addEventListener('click', toggleSidebarMode);
  }

  // Hover slide-out controller: push main content synchronously so it never overlaps
  if (sidebar) {
    sidebar.addEventListener('mouseenter', () => {
      if (document.body.classList.contains('sidebar-collapsed')) {
        document.body.classList.add('sidebar-hovered');
      }
    });

    sidebar.addEventListener('mouseleave', () => {
      document.body.classList.remove('sidebar-hovered');
    });
  }

  // Mobile drawer slide-out toggle

  if (mobileToggleBtn && sidebar) {
    mobileToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('show');
    });

    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('show') && !sidebar.contains(e.target) && !mobileToggleBtn.contains(e.target)) {
        sidebar.classList.remove('show');
      }
    });
  }
}


/* ================================================================
   2. NOTIFICATION SYSTEM (Task 5)
   ================================================================ */
function initNotificationSystem() {
  const bellBtn = document.getElementById('notifBellBtn');
  const dropdown = document.getElementById('notifDropdown');
  const badgeCount = document.getElementById('notifBadgeCount');
  const notifList = document.getElementById('notifList');
  const markAllBtn = document.getElementById('markAllReadBtn');

  if (!bellBtn || !dropdown) return;

  // Track previous unread count to detect NEW notifications
  let _prevUnreadCount = -1;
  let _toastContainer = null;
  let _shownToastIds = new Set();

  // ── Toast helpers ────────────────────────────────────────────────
  function ensureToastContainer() {
    if (!_toastContainer) {
      _toastContainer = document.createElement('div');
      _toastContainer.id = 'dg-toast-container';
      Object.assign(_toastContainer.style, {
        position: 'fixed', bottom: '96px', right: '24px',
        zIndex: '3000', display: 'flex', flexDirection: 'column',
        gap: '10px', pointerEvents: 'none', alignItems: 'flex-end'
      });
      document.body.appendChild(_toastContainer);
    }
    return _toastContainer;
  }

  function showNotifToast(notif) {
    if (_shownToastIds.has(notif.id)) return;
    _shownToastIds.add(notif.id);

    const typeIcons = { analysis: '🔬', system: '🛡️', alert: '⚠️', success: '✅', warning: '⚡' };
    const icon = typeIcons[notif.type] || '🔔';
    const container = ensureToastContainer();

    const wrapper = document.createElement('div');
    wrapper.setAttribute('data-dg-toast', notif.id);
    wrapper.style.cssText = 'pointer-events:auto;animation:toastSlideIn 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards;';

    wrapper.innerHTML = `
      <div style="
        display:flex;align-items:flex-start;gap:12px;
        background:var(--surface);
        border:1px solid rgba(34,211,238,0.3);
        border-radius:16px;padding:14px 16px;
        min-width:300px;max-width:360px;
        box-shadow:0 12px 40px rgba(0,0,0,0.5),0 0 0 1px rgba(34,211,238,0.08) inset;
        backdrop-filter:blur(12px);cursor:pointer;position:relative;overflow:hidden;
      ">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00E5FF,#22D3EE,#0EA5E9);"></div>
        <div style="font-size:22px;line-height:1;flex-shrink:0;margin-top:2px;">${icon}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-weight:700;font-size:13.5px;color:var(--text);margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(notif.title)}</div>
          <div style="font-size:12.5px;color:var(--text-muted);line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">${escapeHtml(notif.message)}</div>
          <div style="font-size:11px;color:rgba(34,211,238,0.7);margin-top:5px;">Tap to view →</div>
        </div>
        <button data-close-toast style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:18px;padding:0;line-height:1;flex-shrink:0;opacity:0.7;" title="Dismiss">×</button>
      </div>`;

    // Click to navigate
    wrapper.addEventListener('click', (e) => {
      if (e.target.closest('[data-close-toast]')) { dismissToast(wrapper); return; }
      markNotifRead(notif.id, notif.link || '');
      dismissToast(wrapper);
    });

    container.prepend(wrapper);

    // Auto-dismiss after 5s
    const timer = setTimeout(() => dismissToast(wrapper), 5000);
    wrapper._dismissTimer = timer;
  }

  function dismissToast(wrapper) {
    clearTimeout(wrapper._dismissTimer);
    wrapper.style.animation = 'toastSlideOut 0.3s ease forwards';
    setTimeout(() => wrapper.remove(), 300);
  }

  // ── Fetch & poll ─────────────────────────────────────────────────
  fetchNotifications();
  // Poll every 30 seconds for new notifications
  setInterval(fetchNotifications, 30000);

  // Toggle Dropdown
  bellBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('show');
    if (dropdown.classList.contains('show')) {
      fetchNotifications();
    }
  });

  document.addEventListener('click', (e) => {
    if (dropdown.classList.contains('show') && !dropdown.contains(e.target) && !bellBtn.contains(e.target)) {
      dropdown.classList.remove('show');
    }
  });

  // Mark all as read
  if (markAllBtn) {
    markAllBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      try {
        const res = await fetch('/notifications/read-all', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          fetchNotifications();
        }
      } catch (err) {
        console.error('Error marking notifications as read:', err);
      }
    });
  }

  async function fetchNotifications() {
    try {
      const res = await fetch('/notifications');
      if (!res.ok) return;
      const data = await res.json();

      // ── Show toast for newly arrived notifications ────────────────
      if (_prevUnreadCount >= 0 && data.unread_count > _prevUnreadCount) {
        const unread = (data.notifications || []).filter(n => !n.is_read);
        const diff = data.unread_count - _prevUnreadCount;
        unread.slice(0, diff).forEach(n => showNotifToast(n));
      }
      _prevUnreadCount = data.unread_count;

      // Update badge
      if (badgeCount) {
        if (data.unread_count > 0) {
          badgeCount.textContent = data.unread_count > 99 ? '99+' : data.unread_count;
          badgeCount.style.display = 'flex';
        } else {
          badgeCount.style.display = 'none';
        }
      }

      // Populate list
      if (notifList) {
        if (!data.notifications || data.notifications.length === 0) {
          notifList.innerHTML = '<div class="notif-empty">No notifications yet</div>';
          return;
        }

        notifList.innerHTML = data.notifications.map(item => `
          <div class="notif-item ${!item.is_read ? 'unread' : ''}" onclick="markNotifRead(${item.id}, '${item.link || ''}')">
            <div class="notif-item-dot"></div>
            <div class="notif-item-content">
              <div class="notif-item-title">${escapeHtml(item.title)}</div>
              <div class="notif-item-msg">${escapeHtml(item.message)}</div>
              <div class="notif-item-time">${escapeHtml(item.created_at)}</div>
            </div>
          </div>
        `).join('');
      }
    } catch (err) {
      console.error('Failed to load notifications:', err);
    }
  }
}

async function markNotifRead(id, targetLink) {
  try {
    await fetch(`/notifications/${id}/read`, { method: 'POST' });
  } catch (e) {}
  if (targetLink) {
    window.location.href = targetLink;
  } else {
    location.reload();
  }
}

/* ================================================================
   3. ANALYZE PAGE HANDLING (Task 2)
   =============================================================== */
function initAnalyzeForm() {
  const form = document.getElementById('analyzeForm');
  const newsText = document.getElementById('newsText');
  const charCounter = document.getElementById('charCounter');
  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const previewBox = document.getElementById('previewBox');
  const previewImg = document.getElementById('previewImg');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  const removeImgBtn = document.getElementById('removeImgBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const inlineError = document.getElementById('inlineError');
  const progressBox = document.getElementById('progressBox');

  if (!form) return;

  // Character Counter (live)
  const MAX_CHARS = 1200;
  const MIN_CHARS = 20;
  if (newsText && charCounter) {
    newsText.addEventListener('input', () => {
      const len = newsText.value.length;
      charCounter.textContent = `${len} / ${MAX_CHARS} characters`;

      // Colour warning when close to limit or under minimum
      if (len >= MAX_CHARS) {
        charCounter.style.color = 'var(--danger, #ef4444)';
      } else if (len > 0 && len < MIN_CHARS) {
        charCounter.style.color = 'var(--warning, #f59e0b)';
      } else {
        charCounter.style.color = '';
      }
    });
  }

  // Image Drag & Drop & File Select
  if (dropzone && imageInput) {
    dropzone.addEventListener('click', (e) => {
      if (e.target !== removeImgBtn && !removeImgBtn.contains(e.target)) {
        imageInput.click();
      }
    });

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        imageInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    imageInput.addEventListener('change', () => {
      if (imageInput.files && imageInput.files[0]) {
        handleFileSelect(imageInput.files[0]);
      }
    });

    if (removeImgBtn) {
      removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.value = '';
        if (previewBox) previewBox.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (fileNameDisplay) fileNameDisplay.textContent = '';
        hideImgBanner();
      });
    }
  }

  function handleFileSelect(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      showError('Invalid file format. Please upload JPG, PNG, or WEBP images.');
      imageInput.value = '';
      hideImgBanner();
      return;
    }

    const MAX_SIZE_MB = 10;
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      const actualSizeMB = (file.size / (1024 * 1024)).toFixed(2);
      showError(`File size (${actualSizeMB} MB) exceeds maximum allowed limit of ${MAX_SIZE_MB} MB. Please upload an image smaller than 10 MB.`);
      imageInput.value = '';
      if (previewBox) previewBox.style.display = 'none';
      if (fileNameDisplay) fileNameDisplay.textContent = '';
      hideImgBanner();
      return;
    }

    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    if (fileNameDisplay) {
      fileNameDisplay.textContent = `${file.name} (${sizeMB} MB)`;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewBox) previewBox.style.display = 'inline-block';
    };
    reader.readAsDataURL(file);
    hideError();

    // Show the image ready feedback banner
    const banner = document.getElementById('imgReadyBanner');
    const bannerTitle = document.getElementById('imgReadyTitle');
    const bannerDetail = document.getElementById('imgReadyDetail');
    if (banner) {
      if (bannerTitle) bannerTitle.textContent = `"${file.name}" ready for analysis`;
      if (bannerDetail) bannerDetail.textContent = `${sizeMB} MB • Cross-modal verification will be activated`;
      banner.style.display = 'flex';
      banner.style.animation = 'fadeInUp 0.35s ease forwards';
    }
  }

  function hideImgBanner() {
    const banner = document.getElementById('imgReadyBanner');
    if (banner) banner.style.display = 'none';
  }

  // Form Submit Handler & Progress Indicator
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    const text = newsText ? newsText.value.trim() : '';
    const file = imageInput && imageInput.files ? imageInput.files[0] : null;

    if (!text && !file) {
      showError('Please provide text content, an image, or both for analysis.');
      return;
    }

    // Enforce minimum 20 characters when text is provided
    if (text) {
      if (text.length < 20) {
        showError(`Text is too short (${text.length} character${text.length === 1 ? '' : 's'} entered). Please enter at least 20 characters for accurate analysis.`);
        if (analyzeBtn) analyzeBtn.disabled = false;
        return;
      }
      // Also guard against going over 1200 characters (belt-and-suspenders)
      if (text.length > 1200) {
        showError('Text exceeds the maximum limit of 1200 characters. Please shorten your input.');
        if (analyzeBtn) analyzeBtn.disabled = false;
        return;
      }
    }

    // Enforce maximum 10 MB image size when an image is selected
    if (file && file.size > 10 * 1024 * 1024) {
      const actualSizeMB = (file.size / (1024 * 1024)).toFixed(2);
      showError(`File size (${actualSizeMB} MB) exceeds maximum allowed limit of 10 MB. Please upload an image smaller than 10 MB.`);
      if (analyzeBtn) analyzeBtn.disabled = false;
      return;
    }

    // Disable button & Show Progress
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (progressBox) progressBox.style.display = 'block';

    const steps = [
      document.getElementById('step1'),
      document.getElementById('step2'),
      document.getElementById('step3'),
      document.getElementById('step4')
    ];

    function setStep(idx) {
      steps.forEach((st, i) => {
        if (!st) return;
        if (i < idx) {
          st.className = 'progress-step-item done';
          st.querySelector('.step-status').textContent = '✓ Done';
        } else if (i === idx) {
          st.className = 'progress-step-item active';
          st.querySelector('.step-status').textContent = 'In progress...';
        } else {
          st.className = 'progress-step-item';
          st.querySelector('.step-status').textContent = 'Waiting';
        }
      });
    }

    setStep(0);
    setTimeout(() => setStep(1), 600);
    setTimeout(() => setStep(2), 1400);

    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('image', file);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error || 'Analysis failed. Please try again.');
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (progressBox) progressBox.style.display = 'none';
        return;
      }

      setStep(3);

      // Brief pause so user sees "Done" state before redirect
      await new Promise(r => setTimeout(r, 700));

      // Redirect to full report page — ?new=1 triggers analysis-complete toast
      if (data.prediction_id) {
        window.location.href = `/report/${data.prediction_id}?new=1`;
      } else {
        location.reload();
      }

    } catch (err) {
      showError('Network error during analysis. Please check server connection.');
      if (analyzeBtn) analyzeBtn.disabled = false;
      if (progressBox) progressBox.style.display = 'none';
    }
  });

  function showError(msg) {
    if (inlineError) {
      inlineError.textContent = msg;
      inlineError.style.display = 'block';
    }
  }

  function hideError() {
    if (inlineError) {
      inlineError.textContent = '';
      inlineError.style.display = 'none';
    }
  }
}

/* ================================================================
   4. REGISTER FORM INLINE VALIDATION & STRENGTH METER (Task 7)
   ================================================================ */
function initRegisterFormValidation() {
  const regForm = document.getElementById('registerForm');
  const emailInput = document.getElementById('regEmail');
  const passInput = document.getElementById('regPassword');
  const confirmInput = document.getElementById('regConfirmPassword');
  const strengthFill = document.getElementById('strengthFill');
  const strengthText = document.getElementById('strengthText');
  const passError = document.getElementById('passMatchError');
  const emailError = document.getElementById('emailError');
  const togglePassBtn = document.getElementById('togglePassBtn');

  if (!regForm) return;

  // Toggle password visibility
  if (togglePassBtn && passInput) {
    togglePassBtn.addEventListener('click', () => {
      const type = passInput.getAttribute('type') === 'password' ? 'text' : 'password';
      passInput.setAttribute('type', type);
      if (confirmInput) confirmInput.setAttribute('type', type);
      togglePassBtn.textContent = type === 'password' ? '👁' : '🙈';
    });
  }

  // Email format check
  if (emailInput && emailError) {
    emailInput.addEventListener('input', () => {
      const val = emailInput.value.trim();
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (val && !emailRegex.test(val)) {
        emailError.textContent = 'Please enter a valid email address (e.g. user@example.com)';
      } else {
        emailError.textContent = '';
      }
    });
  }

  // Password strength meter
  if (passInput && strengthFill && strengthText) {
    passInput.addEventListener('input', () => {
      const val = passInput.value;
      const strength = calcPasswordStrength(val);
      
      strengthFill.style.width = `${strength.score}%`;
      strengthFill.style.backgroundColor = strength.color;
      strengthText.textContent = strength.label;
      strengthText.style.color = strength.color;

      checkPasswordMatch();
    });
  }

  // Confirm password match check
  if (confirmInput) {
    confirmInput.addEventListener('input', checkPasswordMatch);
  }

  function checkPasswordMatch() {
    if (!passInput || !confirmInput || !passError) return;
    if (confirmInput.value && passInput.value !== confirmInput.value) {
      passError.textContent = 'Passwords do not match';
    } else {
      passError.textContent = '';
    }
  }

  function calcPasswordStrength(pass) {
    if (!pass) return { score: 0, label: '', color: 'transparent' };
    let score = 0;
    if (pass.length >= 8) score += 35;
    if (/[A-Z]/.test(pass)) score += 20;
    if (/[0-9]/.test(pass)) score += 20;
    if (/[^A-Za-z0-9]/.test(pass)) score += 25;

    if (score < 40) {
      return { score: Math.max(score, 15), label: 'Weak', color: 'var(--danger)' };
    } else if (score < 75) {
      return { score, label: 'Medium', color: 'var(--warning)' };
    } else {
      return { score: 100, label: 'Strong', color: 'var(--success)' };
    }
  }
}

/* ================================================================
   5. HISTORY PAGE FILTERS & MODAL (Task 4)
   ================================================================ */
function initHistoryFilters() {
  const searchInput = document.getElementById('historySearch');
  const verdictFilter = document.getElementById('verdictFilter');
  const historyTable = document.getElementById('historyTable');

  if (!historyTable) return;

  const rows = Array.from(historyTable.querySelectorAll('tbody tr'));
  const emptyRow = document.getElementById('historyEmptyRow');

  function filterHistory() {
    const q = searchInput ? searchInput.value.toLowerCase().trim() : '';
    const v = verdictFilter ? verdictFilter.value.toUpperCase() : 'ALL';
    let visibleCount = 0;

    rows.forEach(row => {
      if (row.id === 'historyEmptyRow') return;
      const text = row.textContent.toLowerCase();
      const verdict = row.getAttribute('data-verdict') || '';
      const isSaved = row.getAttribute('data-saved') === 'true';

      const matchesQ = !q || text.includes(q);
      const matchesV = v === 'ALL' || (v === 'SAVED' ? isSaved : verdict.includes(v));

      if (matchesQ && matchesV) {
        row.style.display = '';
        visibleCount++;
      } else {
        row.style.display = 'none';
      }
    });

    if (emptyRow) {
      emptyRow.style.display = visibleCount === 0 ? '' : 'none';
    }
  }

  if (searchInput) searchInput.addEventListener('input', filterHistory);
  if (verdictFilter) verdictFilter.addEventListener('change', filterHistory);
}

// Global modal delete trigger
function confirmDeletePrediction(id) {
  const modal = document.getElementById('deleteModal');
  const form = document.getElementById('deleteForm');
  if (modal && form) {
    form.action = `/history/delete/${id}`;
    modal.classList.add('show');
  }
}

function closeDeleteModal() {
  const modal = document.getElementById('deleteModal');
  if (modal) modal.classList.remove('show');
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, function(m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m];
  });
}