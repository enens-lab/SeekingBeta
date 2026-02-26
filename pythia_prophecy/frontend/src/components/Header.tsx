import { MouseEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';

function Header() {
  const navigate = useNavigate();
  const { isAuthenticated, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const scrollToSection = (e: MouseEvent<HTMLAnchorElement>, sectionId: string) => {
    e.preventDefault();
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="site-header">
      <div className="header-content">
        <Link to="/" className="logo">
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta</span>
        </Link>

        <nav className="nav">
          <a
            href="#predictions"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'predictions')}
          >
            Signals
          </a>
          <a
            href="#performance"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'performance')}
          >
            Track Record
          </a>
          <a
            href="#features"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'features')}
          >
            How It Works
          </a>
          <Link to="/pricing" className="nav-link">
            Pricing
          </Link>
        </nav>

        <div className="header-actions">
          <ThemeToggle />
          {isAuthenticated ? (
            <>
              <Link to="/dashboard" className="btn btn-ghost">
                Dashboard
              </Link>
              <button className="btn btn-ghost" onClick={handleLogout}>
                Log Out
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn btn-ghost">
                Log In
              </Link>
              <Link to="/signup" className="btn btn-primary">
                Get Started
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
