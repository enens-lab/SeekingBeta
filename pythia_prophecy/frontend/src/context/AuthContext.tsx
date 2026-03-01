import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth, setToken, getToken, setOnAuthError, User, SignupData } from '../api/client';
import { clearAnalyticsUser, setAnalyticsUser, trackEvent } from '../lib/analytics';

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
  isVerified: boolean;
  login: (email: string, password: string) => Promise<User>;
  signup: (data: SignupData) => Promise<{ message: string }>;
  verifyEmail: (token: string) => Promise<User>;
  resendVerification: (email: string) => Promise<{ message: string }>;
  logout: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  // Register auth error handler for 401 responses (JWT expired)
  useEffect(() => {
    setOnAuthError(() => {
      setUser(null);
      navigate('/login?expired=1', { replace: true });
    });

    return () => {
      setOnAuthError(null);
    };
  }, [navigate]);

  // Check for existing session on mount
  useEffect(() => {
    const token = getToken();
    if (token) {
      auth.getMe()
        .then((userData) => {
          setUser(userData);
        })
        .catch(() => {
          // Token invalid, clear it
          setToken(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user) {
      setAnalyticsUser(user.id, {
        tier: user.tier,
        email_verified: user.email_verified,
      });
      return;
    }
    clearAnalyticsUser();
  }, [user]);

  const login = useCallback(async (email: string, password: string): Promise<User> => {
    setError(null);
    try {
      const response = await auth.login({ email, password });
      setToken(response.access_token);
      setUser(response.user);
      trackEvent('login', { method: 'password' });
      return response.user;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
      trackEvent('login_failed');
      throw err;
    }
  }, []);

  const signup = useCallback(async (data: SignupData): Promise<{ message: string }> => {
    setError(null);
    try {
      const response = await auth.signup(data);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Signup failed';
      setError(message);
      throw err;
    }
  }, []);

  const verifyEmail = useCallback(async (token: string): Promise<User> => {
    setError(null);
    try {
      const response = await auth.verifyEmail(token);
      setToken(response.access_token);
      setUser(response.user);
      trackEvent('email_verified');
      return response.user;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Email verification failed';
      setError(message);
      trackEvent('email_verification_failed');
      throw err;
    }
  }, []);

  const resendVerification = useCallback(async (email: string): Promise<{ message: string }> => {
    setError(null);
    try {
      const response = await auth.resendVerification(email);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to resend verification';
      setError(message);
      throw err;
    }
  }, []);

  const logout = useCallback(() => {
    trackEvent('logout');
    setToken(null);
    setUser(null);
  }, []);

  const value: AuthContextValue = {
    user,
    loading,
    error,
    isAuthenticated: !!user,
    isVerified: user?.email_verified ?? false,
    login,
    signup,
    verifyEmail,
    resendVerification,
    logout,
    clearError: () => setError(null),
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
