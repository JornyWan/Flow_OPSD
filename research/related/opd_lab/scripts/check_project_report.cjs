// Browser verification of the generated, offline HTML. Test tooling stays in .tools/.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');
const {chromium} = require('../.tools/report-browser/node_modules/playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const report = path.join(root, 'runs/local_expert_report_20260918/report.html');
  const screenshots = path.join(root, 'runs/local_expert_report_20260918/screenshots');
  fs.mkdirSync(screenshots, {recursive: true});
  const browser = await chromium.launch({headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
    env: {...process.env, FONTCONFIG_FILE: path.join(root, '.tools/report-browser/fonts/fonts.conf')}});
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}});
  await context.setOffline(true);
  const page = await context.newPage();
  const errors = [], external = [], checks = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('request', r => { if (/^https?:/.test(r.url())) external.push(r.url()); });
  async function screenshot(name) {
    await page.screenshot({path: path.join(screenshots, name + '.png')});
  }
  async function noOverflow() {
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  }
  try {
    await page.goto(pathToFileURL(report).href, {waitUntil: 'load', timeout: 60000});
    await page.waitForFunction(() => window.reportReady === true, {timeout: 30000});
    await page.addStyleTag({content: 'html{scroll-behavior:auto!important}.bar-fill{transition:none!important}'});
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('main>.section').count(), 10);
    assert.equal(await page.locator('.phase-card').count(), 20);
    await noOverflow();
    await screenshot('01-overview-desktop');
    checks.push('offline file opens; 10 sections and all 20 historical phases render');

    assert.deepEqual(await page.locator('#metric-chart .bar-value').allTextContents(), ['6 / 12', '9 / 12', '6 / 12', '6 / 12']);
    await page.selectOption('#metric-path', 'full');
    assert.deepEqual(await page.locator('#metric-chart .bar-value').allTextContents(), ['6 / 12', '8 / 12', '8 / 12', '7 / 12']);
    await page.selectOption('#metric-cohort', 'all_fresh');
    await page.selectOption('#metric-type', 'harm');
    assert.deepEqual(await page.locator('#metric-chart .bar-value').allTextContents(), ['0 / 32', '3 / 32', '2 / 32', '4 / 32']);
    await page.selectOption('#metric-path', 'single');
    assert.deepEqual(await page.locator('#metric-chart .bar-value').allTextContents(), ['0 / 32', '2 / 32', '2 / 32', '1 / 32']);
    await page.selectOption('#metric-type', 'unknown');
    assert.deepEqual(await page.locator('#metric-chart .bar-value').allTextContents(), ['4 / 32', '2 / 32', '2 / 32', '3 / 32']);
    await page.selectOption('#metric-type', 'complete');
    await page.selectOption('#metric-cohort', 'positive_fresh_seed');
    await page.locator('#results').scrollIntoViewIfNeeded();
    await page.evaluate(() => document.getElementById('results').scrollIntoView());
    await screenshot('02-results-desktop');
    checks.push('path, cohort and metric controls agree with frozen counts, including unknowns and harms');

    assert.deepEqual(await page.locator('#human-chart .bar-value').allTextContents(), ['4 / 7', '5 / 7', '4 / 7', '5 / 7']);
    assert.equal(await page.locator('#disagreement-table tbody tr').count(), 11);
    assert.equal(await page.locator('#preferences-table tbody tr').count(), 12);
    await page.locator('[data-jump="1"]').click();
    assert.equal(await page.locator('#case-select option').count(), 32);
    assert((await page.locator('#case-view').innerText()).includes('H02'));
    await page.selectOption('#case-labels', 'human');
    const teacherCard = page.locator('#case-view .image-card').filter({has: page.locator('header', {hasText: '全量教师监督'})});
    assert((await teacherCard.innerText()).includes('不确定'));
    assert((await teacherCard.innerText()).includes('未填'));
    assert(!(await teacherCard.innerText()).includes('确认完全成功'));
    assert.equal(await page.locator('#case-view .caption').filter({hasText: '助手冻结标签'}).count(), 0);
    await page.selectOption('#case-labels', 'both');
    await page.locator('#case-view img').first().scrollIntoViewIfNeeded();
    await page.waitForFunction(() => [...document.querySelectorAll('#case-view img')].every(im => im.complete && im.naturalWidth === 512));
    await page.evaluate(() => document.getElementById('cases').scrollIntoView());
    await screenshot('03-case-gallery-desktop');
    await page.locator('#case-view img').first().click();
    assert(await page.locator('#image-modal').evaluate(e => e.open));
    await page.waitForFunction(() => document.getElementById('modal-image').naturalWidth === 512);
    await page.keyboard.press('Escape');
    assert(!(await page.locator('#image-modal').evaluate(e => e.open)));
    checks.push('human uncertainty and missing fields remain separate; case jump and original-size image modal work');

    await page.selectOption('#case-path', 'all');
    assert.equal(await page.locator('#case-select option').count(), 64);
    let observations = 0;
    for (const [dataset, groups, cards] of [['evaluation', 64, 4], ['collection', 48, 2], ['teacher', 12, 4]]) {
      await page.selectOption('#case-dataset', dataset);
      assert.equal(await page.locator('#case-path-label').isVisible(), dataset === 'evaluation');
      assert.equal(await page.locator('#case-cohort-label').isVisible(), dataset === 'evaluation');
      assert.equal(await page.locator('#case-select option').count(), groups);
      for (let i = 0; i < groups; i++) {
        await page.selectOption('#case-select', String(i));
        assert.equal(await page.locator('#case-view .image-card').count(), cards);
        const ids = await page.locator('#case-view img').evaluateAll(images => images.map(x => x.dataset.image));
        assert(ids.every(x => /^[a-f0-9]{64}$/.test(x)));
        observations += cards;
      }
    }
    assert.equal(observations, 400);
    checks.push('all 112 gallery groups / 400 observation slots are reachable');

    await page.selectOption('#history-filter', '测量审查');
    assert.equal(await page.locator('.phase-card').count(), 5);
    await page.selectOption('#history-filter', 'all');
    await page.fill('#history-search', 'Phase15');
    // Search includes references in other phases' complete historical records.
    assert(await page.locator('#phase-15').isVisible());
    await page.fill('#history-search', '标签辅助筛选训练');
    assert.equal(await page.locator('.phase-card').count(), 1);
    await page.fill('#history-search', '');
    assert.equal(await page.locator('.phase-card').count(), 20);
    checks.push('historical category and full-record search work without dropping phases');

    await page.locator('#mask-toggle').check();
    assert((await page.locator('#query-total').innerText()).includes('912'));
    await page.locator('#mask-toggle').uncheck();
    assert((await page.locator('#query-total').innerText()).includes('1536'));
    const csvWait = page.waitForEvent('download');
    await page.locator('#download-metrics').click();
    const csv = await csvWait;
    assert.equal(csv.suggestedFilename(), 'phase20-single-positive_fresh_seed.csv');
    const csvText = fs.readFileSync(await csv.path(), 'utf8');
    assert(csvText.includes('assistant_frozen_provisional'));
    assert(csvText.includes('"certified","12","6","1","1"'));
    checks.push('mask illustration preserves counts; CSV exports annotation source and current denominator');

    await page.locator('#present-btn').click();
    assert(await page.locator('body').evaluate(e => e.classList.contains('presenter')));
    await page.evaluate(() => document.getElementById('overview').scrollIntoView());
    await page.keyboard.press('ArrowRight');
    assert(Math.abs(await page.locator('#method').evaluate(e => e.getBoundingClientRect().top)) < 40);
    await page.locator('#present-exit').click();
    assert(!(await page.locator('body').evaluate(e => e.classList.contains('presenter'))));
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => window.scrollTo(0, 0));
    await noOverflow();
    await screenshot('04-overview-mobile');
    await page.selectOption('#case-dataset', 'evaluation');
    await page.selectOption('#case-path', 'single');
    await page.evaluate(() => document.getElementById('cases').scrollIntoView());
    await noOverflow();
    await screenshot('05-gallery-mobile');
    checks.push('presentation toggle and desktop/mobile layouts work without horizontal page overflow');
    await page.setViewportSize({width: 1440, height: 1100});
    await page.evaluate(() => window.prepareReportPrint());
    assert(await page.locator('img[data-image]').evaluateAll(images => images.every(im => im.complete && im.naturalWidth > 0)));
    await page.pdf({path: path.join(root, 'runs/local_expert_report_20260918/print-preview.pdf'), format: 'A4', printBackground: true});
    checks.push('print preparation loads embedded figures; Chromium can export the report to PDF');
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    const result = {report_sha256: crypto.createHash('sha256').update(fs.readFileSync(report)).digest('hex'),
      browser: browser.version(), offline: true, external_requests: external, page_errors: errors,
      checked_observations: observations, checks, screenshots: fs.readdirSync(screenshots).map(f => path.relative(root, path.join(screenshots, f)))};
    fs.writeFileSync(path.join(root, 'experiments/local_expert_report_20260918/browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
