// Authentication handling for Pythia

const AUTH_TOKEN_KEY = 'pythia_token';
const AUTH_USER_KEY = 'pythia_user';

const Auth = {
    getToken() {
        return localStorage.getItem(AUTH_TOKEN_KEY);
    },

    setToken(token) {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
    },

    getUser() {
        const user = localStorage.getItem(AUTH_USER_KEY);
        return user ? JSON.parse(user) : null;
    },

    setUser(user) {
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
    },

    isLoggedIn() {
        return !!this.getToken();
    },

    logout() {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        window.location.reload();
    },

    async login(username, password) {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const res = await fetch('/auth/login', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || 'Login failed');
        }

        const data = await res.json();
        this.setToken(data.access_token);
        this.setUser(data.user);
        return data.user;
    },

    getAuthHeaders() {
        const token = this.getToken();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    },

    async fetchWithAuth(url, options = {}) {
        const headers = {
            ...options.headers,
            ...this.getAuthHeaders()
        };

        const res = await fetch(url, { ...options, headers });

        if (res.status === 401) {
            this.logout();
            throw new Error('Session expired');
        }

        return res;
    }
};

// Initialize auth UI on any page that includes this script
function initAuthUI() {
    const userInfo = document.getElementById('user-info');
    const loginModal = document.getElementById('login-modal');
    const loginForm = document.getElementById('login-form');

    if (!userInfo) return;

    if (Auth.isLoggedIn()) {
        const user = Auth.getUser();
        userInfo.innerHTML = `
            <span class="tier-badge tier-${user.tier}">${user.tier.toUpperCase()}</span>
            <span class="username">${user.username}</span>
            <button id="logout-btn" class="btn btn-secondary">Logout</button>
        `;
        document.getElementById('logout-btn').addEventListener('click', () => Auth.logout());
    } else {
        userInfo.innerHTML = `<button id="login-btn" class="btn">Login</button>`;
        document.getElementById('login-btn').addEventListener('click', () => {
            if (loginModal) {
                loginModal.classList.remove('hidden');
            }
        });
    }

    // Login form handling
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = loginForm.username.value;
            const password = loginForm.password.value;
            const errorEl = document.getElementById('login-error');
            const submitBtn = loginForm.querySelector('button[type="submit"]');

            try {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Logging in...';
                await Auth.login(username, password);
                if (loginModal) loginModal.classList.add('hidden');
                window.location.reload();
            } catch (err) {
                if (errorEl) {
                    errorEl.textContent = err.message;
                    errorEl.classList.remove('hidden');
                }
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Login';
            }
        });
    }

    // Close modal on backdrop click
    if (loginModal) {
        loginModal.addEventListener('click', (e) => {
            if (e.target === loginModal) {
                loginModal.classList.add('hidden');
            }
        });
    }
}

window.addEventListener('DOMContentLoaded', initAuthUI);
