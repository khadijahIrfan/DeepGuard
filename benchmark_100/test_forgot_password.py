import requests

BASE = 'http://127.0.0.1:5000'
s = requests.Session()

print('=== TESTING FORGOT & RESET PASSWORD FLOW ===')

# Step 1: GET /forgot-password
r = s.get(f'{BASE}/forgot-password')
print('1. GET /forgot-password -> Status:', r.status_code)
assert r.status_code == 200

# Step 2: POST unknown email
r_bad = s.post(f'{BASE}/forgot-password', data={'email': 'nobody_exists@gmail.com'})
print('2. Unknown email test -> Contains error:', 'No account registered with that email' in r_bad.text)
assert 'No account registered with that email' in r_bad.text

# Step 3: POST valid registered email
r_good = s.post(f'{BASE}/forgot-password', data={'email': 'khadija@gmail.com'})
print('3. Valid email test -> Contains Reset Link Ready:', 'Reset Link Ready!' in r_good.text)
print('   Contains direct button:', 'Reset Password Now' in r_good.text)
assert 'Reset Password Now' in r_good.text

# Extract token URL
token_url = None
for part in r_good.text.split('href="'):
    if 'reset-password/' in part:
        token_url = part.split('"')[0]
        break

print('4. Extracted reset URL:', token_url)
assert token_url is not None

# Step 4: GET reset page
r_reset_page = s.get(token_url)
print('5. GET reset page -> Status:', r_reset_page.status_code, 'Title:', 'Set New Password' in r_reset_page.text)
assert r_reset_page.status_code == 200

# Step 5: Test mismatched password
r_mismatch = s.post(token_url, data={'password': 'newpassword123', 'confirm_password': 'differentpassword'})
print('6. Password mismatch test -> Error shown:', 'Passwords do not match' in r_mismatch.text)
assert 'Passwords do not match' in r_mismatch.text

# Step 6: Test successful password update
r_update = s.post(token_url, data={'password': 'ResetSuccessPass2026', 'confirm_password': 'ResetSuccessPass2026'}, allow_redirects=True)
print('7. Password update -> Final URL:', r_update.url)
print('   Success flash message:', 'Your password has been reset successfully' in r_update.text)
assert 'Your password has been reset successfully' in r_update.text

# Step 7: Test login with the newly set password
r_login = s.post(f'{BASE}/login', data={'email': 'khadija@gmail.com', 'password': 'ResetSuccessPass2026'}, allow_redirects=True)
print('8. Login with NEW password -> Final URL:', r_login.url)
print('   Logged in to dashboard:', r_login.url.endswith('/dashboard') or r_login.url.endswith('/'))
assert r_login.url.endswith('/dashboard') or r_login.url.endswith('/')

print('\nALL FORGOT-PASSWORD & RESET-PASSWORD TESTS PASSED 100%!')
