const { app, BrowserWindow, dialog, session } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const readline = require('node:readline');
const { setTimeout: delay } = require('node:timers/promises');

const smoke = process.argv.includes('--release-smoke');
let backend;
let window;
let closing = false;
let exitCode = 0;
app.commandLine.appendSwitch('disable-background-networking');

function dataDirectory() {
  if (process.env.ARION_DATA_DIR) return path.resolve(process.env.ARION_DATA_DIR);
  if (process.platform === 'win32') {
    return path.join(process.env.LOCALAPPDATA || app.getPath('appData'), 'ARION');
  }
  if (process.platform === 'darwin') return path.join(app.getPath('appData'), 'ARION');
  return path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'), 'arion');
}

function recordLaunch(message) {
  fs.appendFileSync(path.join(dataDirectory(), 'desktop.log'), `${message}\n`, 'utf8');
}

async function startBackend() {
  const root = path.join(process.resourcesPath, 'arion-core');
  const runtime = path.join(root, 'runtime');
  const python = path.join(runtime, process.platform === 'win32' ? 'python.exe' : 'bin/python3.10');
  const backendDir = path.join(root, 'irion-upload/UI/backend');
  const data = dataDirectory();
  fs.mkdirSync(data, { recursive: true });
  const log = fs.openSync(path.join(data, 'desktop.log'), 'a');
  const { PYTHONPATH, PYTHONHOME, ARION_V02_WHOLE_ROOT, ARION_NEW19_WHOLE_ROOT, ...parent } = process.env;
  try {
    backend = spawn(python, ['-s', '-B', '-m', 'arion.desktop'], {
      cwd: backendDir,
      env: {
        ...parent,
        PYTHONHOME: runtime,
        PYTHONPATH: [path.join(root, 'ai'), path.join(root, 'ai/src'), backendDir].join(path.delimiter),
        PYTHONNOUSERSITE: '1',
        PYTHONUTF8: '1',
        ARION_AI_ROOT: root,
        ARION_AI_PYTHON: python,
        ARION_DATA_DIR: data,
        ARION_V02_INFLIGHT_ROOT: path.join(root, 'models/v02'),
        OMP_NUM_THREADS: '1',
        OPENBLAS_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
      },
      stdio: ['pipe', 'pipe', log],
      windowsHide: true,
    });
  } finally {
    fs.closeSync(log);
  }
  backend.stdin.on('error', () => {});
  const reader = readline.createInterface({ input: backend.stdout });
  const port = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('서버가 수신 주소를 준비하지 못했습니다.')), 10000);
    backend.once('error', reject);
    backend.once('exit', (code) => reject(new Error(`서버가 종료됐습니다 (${code}).`)));
    reader.once('line', (line) => {
      clearTimeout(timer);
      try {
        const { port: value } = JSON.parse(line);
        if (!Number.isInteger(value) || value < 1 || value > 65535) throw new Error('잘못된 수신 포트');
        resolve(value);
      } catch (error) {
        reject(error);
      }
    });
  });
  const url = `http://127.0.0.1:${port}`;
  let lastHealth = '응답 없음';
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (backend.exitCode !== null || backend.signalCode !== null) break;
    try {
      const response = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(1000) });
      const health = await response.json();
      if (health?.edition === 'v0.2' && health.status === 'ok') return url;
      lastHealth = `HTTP ${response.status} · ${health?.edition ?? '버전 확인 실패'}`;
    } catch (error) {
      lastHealth = error.cause?.code || error.message;
    }
    await delay(100);
  }
  throw new Error(`ARION 서버의 준비를 확인하지 못했습니다 (${lastHealth}). ${path.join(data, 'desktop.log')}`);
}

function closeBackend(event) {
  if (!backend?.pid || backend.exitCode !== null || backend.signalCode !== null) {
    if (exitCode) app.exit(exitCode);
    return;
  }
  event.preventDefault();
  if (closing) return;
  closing = true;
  backend.stdin.end();
  const timer = setTimeout(() => backend.kill(), 15000);
  backend.once('exit', () => {
    clearTimeout(timer);
    app.exit(exitCode);
  });
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('before-quit', closeBackend);
  app.on('window-all-closed', () => app.quit());
  app.on('second-instance', () => {
    if (window?.isMinimized()) window.restore();
    window?.focus();
  });
  app.whenReady().then(async () => {
    const url = await startBackend();
    session.defaultSession.setPermissionCheckHandler(() => false);
    session.defaultSession.setPermissionRequestHandler((_contents, _permission, done) => done(false));
    window = new BrowserWindow({
      width: 1280,
      height: 900,
      minWidth: 760,
      show: !smoke,
      autoHideMenuBar: true,
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.webContents.on('will-navigate', (event, target) => {
      if (new URL(target).origin !== url) event.preventDefault();
    });
    await window.loadURL(url);
    if (smoke) {
      const catalog = await fetch(`${url}/api/local-ai/catalog`).then((response) => response.json());
      const title = window.getTitle();
      if (title !== 'ARION · 비행 인사이트' || catalog.models?.length !== 8) {
        throw new Error(`데스크톱 화면 또는 V0.2 모델 확인 실패: ${JSON.stringify({
          title, models: catalog.models?.length,
        })}`);
      }
      const result = JSON.stringify({ desktop: true, models: catalog.models.length });
      recordLaunch(result);
      console.log(result);
      app.quit();
    }
  }).catch((error) => {
    try {
      recordLaunch(error.stack || String(error));
    } catch (logError) {
      console.error(logError);
    }
    if (smoke) {
      console.error(error);
      exitCode = 1;
    } else {
      dialog.showErrorBox('ARION 실행 오류', String(error.message));
    }
    app.quit();
  });
}
