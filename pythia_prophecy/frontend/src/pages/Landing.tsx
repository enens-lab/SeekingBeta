import Header from '../components/Header';
import Hero from '../components/Hero';
import PredictionsCarousel from '../components/PredictionsCarousel';
import SportsPreview from '../components/SportsPreview';
import PerformanceComparison from '../components/PerformanceComparison';
import Features from '../components/Features';
import TrustBox from '../components/TrustBox';
import AppDownload from '../components/AppDownload';
import BetaTesterSignup from '../components/BetaTesterSignup';
import CallToAction from '../components/CallToAction';
import Footer from '../components/Footer';

function Landing() {
  return (
    <>
      <Header />
      <main>
        <Hero />
        <PerformanceComparison />
        <PredictionsCarousel />
        <SportsPreview />
        <Features />
        <TrustBox />
        <AppDownload />
        <BetaTesterSignup />
        <CallToAction />
      </main>
      <Footer />
    </>
  );
}

export default Landing;
