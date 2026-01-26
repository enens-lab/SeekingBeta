function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="site-footer">
      <div className="footer-content">
        <div className="footer-brand">
          <span className="logo-icon">P</span>
          <span className="logo-text">Pythia</span>
        </div>
        <p className="footer-disclaimer">
          Pythia is for educational and research purposes only.
          Past performance does not guarantee future results.
          Always do your own research and consult a financial advisor.
        </p>
        <p className="footer-copyright">
          &copy; {currentYear} Pythia. All rights reserved.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
