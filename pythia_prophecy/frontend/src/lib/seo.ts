export interface SeoMeta {
  title: string;
  description: string;
  canonicalPath?: string;
  indexable: boolean;
}

export const SITE_ORIGIN = 'https://seekingbeta.ai';

export const DEFAULT_SEO: SeoMeta = {
  title: 'SeekingBeta.AI | Prediction Boards For Stocks And Sports',
  description:
    'SeekingBeta.AI publishes model-driven prediction boards for stocks and sports with transparent probabilities and visible track records.',
  canonicalPath: '/',
  indexable: true,
};

export const ROUTE_SEO: Record<string, SeoMeta> = {
  '/': DEFAULT_SEO,
  '/pricing': {
    title: 'Pricing | SeekingBeta.AI',
    description:
      'Compare SeekingBeta.AI plans for stock and sports prediction boards, watchlist limits, and deeper analysis access.',
    canonicalPath: '/pricing',
    indexable: true,
  },
  '/methodology': {
    title: 'Model Methodology | SeekingBeta.AI',
    description:
      'Learn how SeekingBeta.AI builds, evaluates, and presents model-generated probability boards.',
    canonicalPath: '/methodology',
    indexable: true,
  },
  '/track-record': {
    title: 'Track Record | SeekingBeta.AI',
    description:
      'See how SeekingBeta.AI model ratings have performed against the S&P 500 benchmark, with hit rate, Sharpe, and drawdown.',
    canonicalPath: '/track-record',
    indexable: true,
  },
  '/terms': {
    title: 'Terms of Service | SeekingBeta.AI',
    description: 'Review the SeekingBeta.AI terms of service and platform usage terms.',
    canonicalPath: '/terms',
    indexable: true,
  },
  '/privacy': {
    title: 'Privacy Policy | SeekingBeta.AI',
    description: 'Review how SeekingBeta.AI collects, uses, and protects your data.',
    canonicalPath: '/privacy',
    indexable: true,
  },
  '/refund-cancellation': {
    title: 'Refund & Cancellation | SeekingBeta.AI',
    description: 'Read the SeekingBeta.AI refund and cancellation policy for paid subscriptions.',
    canonicalPath: '/refund-cancellation',
    indexable: true,
  },
  '/delete-account': {
    title: 'Delete Your Account | SeekingBeta.AI',
    description: 'Permanently delete your SeekingBeta.AI account and associated data from the web, the app, or by email.',
    canonicalPath: '/delete-account',
    indexable: true,
  },
  '/support': {
    title: 'Support | SeekingBeta.AI',
    description: 'Get help with your SeekingBeta.AI account, billing, and prediction boards.',
    canonicalPath: '/support',
    indexable: true,
  },
  '/login': {
    title: 'Log In | SeekingBeta.AI',
    description: 'Log in to access your SeekingBeta.AI prediction boards.',
    canonicalPath: '/login',
    indexable: false,
  },
  '/signup': {
    title: 'Sign Up | SeekingBeta.AI',
    description: 'Create your SeekingBeta.AI account and unlock free prediction boards.',
    canonicalPath: '/signup',
    indexable: false,
  },
  '/verify-email': {
    title: 'Verify Email | SeekingBeta.AI',
    description: 'Verify your email to activate your SeekingBeta.AI account.',
    canonicalPath: '/verify-email',
    indexable: false,
  },
  '/reset-password': {
    title: 'Reset Password | SeekingBeta.AI',
    description: 'Reset your SeekingBeta.AI account password.',
    canonicalPath: '/reset-password',
    indexable: false,
  },
  '/dashboard': {
    title: 'Dashboard | SeekingBeta.AI',
    description: 'Your personalized stock prediction board and watchlist views.',
    canonicalPath: '/dashboard',
    indexable: false,
  },
  '/sports': {
    title: 'Sports Predictions | SeekingBeta.AI',
    description: 'Market-style probability boards for PGA, LPGA, ATP, and WTA without betting or trading.',
    canonicalPath: '/sports',
    indexable: false,
  },
  '/oracle': {
    title: 'Watchlist | SeekingBeta.AI',
    description: 'Manage your watchlist board and preferred model horizons.',
    canonicalPath: '/oracle',
    indexable: false,
  },
  '/analysis': {
    title: 'Analysis | SeekingBeta.AI',
    description: 'Run board scans, compare probabilities, and review model output.',
    canonicalPath: '/analysis',
    indexable: false,
  },
  '/profile': {
    title: 'Profile | SeekingBeta.AI',
    description: 'Manage your account profile and billing.',
    canonicalPath: '/profile',
    indexable: false,
  },
};

export function getRouteSeo(pathname: string): SeoMeta {
  return ROUTE_SEO[pathname] ?? {
    ...DEFAULT_SEO,
    canonicalPath: pathname || '/',
  };
}

// Public, indexable routes that get a static prerendered index.html at build
// time (scripts/prerender.mjs). Keep in sync with public/sitemap.xml.
export const PRERENDER_ROUTES = Object.entries(ROUTE_SEO)
  .filter(([, seo]) => seo.indexable)
  .map(([path]) => path);
