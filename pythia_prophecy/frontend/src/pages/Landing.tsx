import Header from '../components/Header';
import Hero from '../components/Hero';
import PerformanceComparison from '../components/PerformanceComparison';
import PredictionsCarousel from '../components/PredictionsCarousel';
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
