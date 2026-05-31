import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
  const dims = await page.evaluate(() => {
    const el = document.querySelector('.dashboard-layout');
    return el ? { width: el.clientWidth, height: el.clientHeight } : null;
  });
  console.log('Dashboard layout dimensions:', dims);
  await browser.close();
})();
