import { FormEvent, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import { auth, billing, BillingStatus } from '../api/client';

function Profile() {
  const navigate = useNavigate();
  const { user, logout, isVerified } = useAuth();
  const toast = useToast();

  const [billingStatus, setBillingStatus] = useState<BillingStatus | null>(null);
  const [billingLoading, setBillingLoading] = useState(false);
  const [portalLoading, setPortalLoading] = useState(false);
  const [cancelLoading, setCancelLoading] = useState(false);
  const [changeLoading, setChangeLoading] = useState(false);
  const [targetTier, setTargetTier] = useState<'basic' | 'pro'>('basic');

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordLoading, setPasswordLoading] = useState(false);

  const [deletePassword, setDeletePassword] = useState('');
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteLoading, setDeleteLoading] = useState(false);

  const loadBillingStatus = async () => {
    if (!isVerified) {
      setBillingStatus(null);
      return;
    }
    setBillingLoading(true);
    try {
      const status = await billing.getStatus();
      setBillingStatus(status);
      const effective = status.effective_tier === 'pro' ? 'pro' : 'basic';
      setTargetTier(effective);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load billing status';
      toast.error(message);
    } finally {
      setBillingLoading(false);
    }
  };

  useEffect(() => {
    loadBillingStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVerified]);

  const handleChangePassword = async (event: FormEvent) => {
    event.preventDefault();
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error('Please complete all password fields.');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('New password and confirmation do not match.');
      return;
    }

    setPasswordLoading(true);
    try {
      const result = await auth.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      toast.success(result.message || 'Password updated.');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update password';
      toast.error(message);
    } finally {
      setPasswordLoading(false);
    }
  };

  const handleOpenBillingPortal = async () => {
    setPortalLoading(true);
    try {
      const result = await billing.createPortalSession();
      window.location.assign(result.portal_url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to open billing portal';
      toast.error(message);
    } finally {
      setPortalLoading(false);
    }
  };

  const handleCancelSubscription = async () => {
    if (!window.confirm('Cancel your subscription at the end of the current billing period?')) {
      return;
    }
    setCancelLoading(true);
    try {
      const result = await billing.cancelSubscription();
      toast.success(result.message || 'Subscription cancellation scheduled.');
      await loadBillingStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel subscription';
      toast.error(message);
    } finally {
      setCancelLoading(false);
    }
  };

  const handleChangeSubscription = async () => {
    setChangeLoading(true);
    try {
      const result = await billing.changeSubscription(targetTier);
      if (result.mode === 'checkout' && result.checkout_url) {
        window.location.assign(result.checkout_url);
        return;
      }
      toast.success(result.message || 'Subscription updated.');
      await loadBillingStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to change subscription';
      toast.error(message);
    } finally {
      setChangeLoading(false);
    }
  };

  const handleDeleteAccount = async (event: FormEvent) => {
    event.preventDefault();
    if (!window.confirm('Delete your account permanently? This action cannot be undone.')) {
      return;
    }
    setDeleteLoading(true);
    try {
      const result = await auth.deleteAccount({
        password: deletePassword,
        confirm_text: deleteConfirmText,
      });
      toast.success(result.message || 'Account deleted.');
      logout();
      navigate('/', { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete account';
      toast.error(message);
    } finally {
      setDeleteLoading(false);
    }
  };

  const subscriptionCancelable = Boolean(
    billingStatus &&
      billingStatus.billing_enabled &&
      ['active', 'trialing', 'past_due'].includes((billingStatus.subscription_status || '').toLowerCase()) &&
      !billingStatus.cancel_at_period_end
  );

  return (
    <div className="dashboard-page">
      <DashboardHeader activePage="profile" />

      <main className="dashboard-main">
        <div className="profile-page">
          <section className="profile-section">
            <h1>Account Profile</h1>
            <div className="profile-meta">
              <p><strong>Name:</strong> {user?.first_name} {user?.last_name}</p>
              <p><strong>Email:</strong> {user?.email}</p>
              <p><strong>Current Tier:</strong> {(billingStatus?.effective_tier || user?.tier || 'free').toUpperCase()}</p>
            </div>
          </section>

          <section className="profile-section">
            <h2>Change Password</h2>
            <form className="profile-form" onSubmit={handleChangePassword}>
              <label>
                Current password
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </label>
              <label>
                New password
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </label>
              <label>
                Confirm new password
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </label>
              <button type="submit" className="btn btn-primary" disabled={passwordLoading}>
                {passwordLoading ? 'Updating...' : 'Update Password'}
              </button>
            </form>
          </section>

          <section className="profile-section">
            <h2>Subscription</h2>
            {!isVerified && (
              <p className="profile-note">Verify your email to manage subscription settings.</p>
            )}
            {isVerified && (
              <>
                {billingLoading ? (
                  <p className="profile-note">Loading billing status...</p>
                ) : (
                  <div className="profile-meta">
                    <p><strong>Status:</strong> {billingStatus?.subscription_status || 'none'}</p>
                    <p><strong>Cancel at period end:</strong> {billingStatus?.cancel_at_period_end ? 'Yes' : 'No'}</p>
                    <p>
                      <strong>Current period end:</strong>{' '}
                      {billingStatus?.current_period_end
                        ? new Date(billingStatus.current_period_end).toLocaleString()
                        : 'N/A'}
                    </p>
                  </div>
                )}
                <div className="profile-actions">
                  <label className="profile-tier-select">
                    <span>Switch plan</span>
                    <select
                      value={targetTier}
                      onChange={(e) => setTargetTier(e.target.value as 'basic' | 'pro')}
                      disabled={changeLoading || billingLoading}
                    >
                      <option value="basic">Basic ($9.99/month)</option>
                      <option value="pro">Pro ($19.99/month)</option>
                    </select>
                  </label>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleChangeSubscription}
                    disabled={changeLoading || billingLoading}
                  >
                    {changeLoading ? 'Updating...' : 'Change Subscription'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline"
                    onClick={handleOpenBillingPortal}
                    disabled={portalLoading || billingLoading}
                  >
                    {portalLoading ? 'Opening...' : 'Manage Billing'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline"
                    onClick={handleCancelSubscription}
                    disabled={!subscriptionCancelable || cancelLoading || billingLoading}
                  >
                    {cancelLoading ? 'Canceling...' : 'Cancel Subscription'}
                  </button>
                </div>
              </>
            )}
          </section>

          <section className="profile-section profile-danger">
            <h2>Delete Account</h2>
            <p className="profile-note">
              This deletes your account permanently. Type <code>DELETE</code> to confirm.
            </p>
            <form className="profile-form" onSubmit={handleDeleteAccount}>
              <label>
                Password
                <input
                  type="password"
                  value={deletePassword}
                  onChange={(e) => setDeletePassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </label>
              <label>
                Confirmation text
                <input
                  type="text"
                  value={deleteConfirmText}
                  onChange={(e) => setDeleteConfirmText(e.target.value)}
                  placeholder="DELETE"
                  required
                />
              </label>
              <button type="submit" className="btn btn-danger" disabled={deleteLoading}>
                {deleteLoading ? 'Deleting...' : 'Delete Account'}
              </button>
            </form>
          </section>
        </div>
      </main>
    </div>
  );
}

export default Profile;
