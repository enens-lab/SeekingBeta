import Header from '../components/Header';
import Hero from '../components/Hero';
import PredictionsCarousel from '../components/PredictionsCarousel';
import PerformanceComparison from '../components/PerformanceComparison';
import Features from '../components/Features';
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
        <Features />
        <CallToAction />
      </main>
      <Footer />
    </>
  );
}

export default Landing;
