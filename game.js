/* ============================================================
   Arcade – 5 games in one file
   Games: NYC Chase | Snake | Asteroids | Breakout | Flappy Bird
   ============================================================ */
'use strict';

// Shared state
let canvas, ctx, raf, gameRunning;
let currentGame = null;
let score = 0, lives = 3, level = 1, frameCount = 0;
let keys = {};
function onKey(e)  { keys[e.key] = true;  e.preventDefault(); }
function offKey(e) { keys[e.key] = false; }
function getCanvas() { canvas = document.getElementById('game-canvas'); ctx = canvas.getContext('2d'); }

// Public API
function showLauncher() {
  document.getElementById('game-launcher').classList.remove('hidden');
  document.getElementById('game-area').classList.add('hidden');
}
function launchGame(id) { stopGame(); currentGame = id; startCurrentGame(); }
function startCurrentGame() {
  if      (currentGame === 'chase')     startChase();
  else if (currentGame === 'snake')     startSnake();
  else if (currentGame === 'asteroids') startAsteroids();
  else if (currentGame === 'breakout')  startBreakout();
  else if (currentGame === 'flappy')    startFlappy();
  else if (currentGame === 'catrun')    startCatRun();
}
function stopGame() {
  gameRunning = false;
  cancelAnimationFrame(raf);
  document.removeEventListener('keydown', onKey);
  document.removeEventListener('keyup',   offKey);
  if (canvas) canvas.removeEventListener('mousemove', brkMouse);
  if (canvas) canvas.removeEventListener('click', flap);
  document.removeEventListener('keydown', snakeKey);
  document.removeEventListener('keydown', flappyKey);
  keys = {};
}
function gameOver(emoji, sub) {
  stopGame();
  document.getElementById('msg-title').textContent = emoji + ' Game Over';
  document.getElementById('msg-sub').textContent   = sub || 'Score: ' + score;
  document.getElementById('game-msg').classList.remove('hidden');
}
function updateHUD() {
  document.getElementById('hud-score').textContent = 'Score: ' + score;
  document.getElementById('hud-level').textContent = 'Level: ' + level;
  const h = ['\u{1F480}','\u2764\uFE0F','\u2764\uFE0F\u2764\uFE0F','\u2764\uFE0F\u2764\uFE0F\u2764\uFE0F'];
  document.getElementById('hud-lives').textContent = h[Math.max(0,Math.min(3,lives))];
}

// =============================================================
//  GAME 1 – NYC CHASE
// =============================================================
const TILE=40,COLS=17,ROWS=15,CW=COLS*TILE,CH=ROWS*TILE;
const MAP=[
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,2,2,2,0,1,1,1,0,2,2,2,0,1,1],
  [1,1,0,2,2,2,0,1,1,1,0,2,2,2,0,1,1],
  [1,1,0,2,2,2,0,1,1,1,0,2,2,2,0,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,2,2,2,0,1,1,1,0,2,2,2,0,1,1],
  [1,1,0,2,2,2,0,1,1,1,0,2,2,2,0,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
];
function isRoad(c,r){return c>=0&&r>=0&&c<COLS&&r<ROWS&&MAP[r][c]===0;}
let thief,police,particles=[];
function startChase(){
  getCanvas(); canvas.width=CW; canvas.height=CH;
  score=0;lives=3;level=1;frameCount=0;particles=[];
  thief={x:2*TILE+TILE/2,y:2*TILE+TILE/2,angle:0,vx:0,vy:0,baseSpeed:2.8};
  spawnPolice(1);
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning=true;
  document.addEventListener('keydown',onKey);
  document.addEventListener('keyup',offKey);
  (function loop(){if(!gameRunning)return;raf=requestAnimationFrame(loop);frameCount++;chaseUpdate();chaseDraw();updateHUD();})();
}
function spawnPolice(n){
  police=[];
  const sp=[{c:14,r:2},{c:14,r:10},{c:2,r:10},{c:8,r:6},{c:2,r:6}];
  for(let i=0;i<n;i++){const s=sp[i%sp.length];police.push({x:s.c*TILE+TILE/2,y:s.r*TILE+TILE/2,angle:0,speed:1.6+level*0.25,pathTimer:0,targetX:0,targetY:0});}
}
function canMoveC(x,y,h){return [{cx:x-h,cy:y-h},{cx:x+h,cy:y-h},{cx:x-h,cy:y+h},{cx:x+h,cy:y+h}].every(({cx,cy})=>isRoad(Math.floor(cx/TILE),Math.floor(cy/TILE)));}
function chaseUpdate(){
  const t=thief,sp=t.baseSpeed;let ax=0,ay=0;
  if(keys['ArrowLeft']||keys['a']||keys['A'])ax=-sp;
  if(keys['ArrowRight']||keys['d']||keys['D'])ax=sp;
  if(keys['ArrowUp']||keys['w']||keys['W'])ay=-sp;
  if(keys['ArrowDown']||keys['s']||keys['S'])ay=sp;
  if(ax&&ay){ax*=0.707;ay*=0.707;}
  if(canMoveC(t.x+ax,t.y,14))t.x+=ax;
  if(canMoveC(t.x,t.y+ay,14))t.y+=ay;
  if(ax||ay){t.angle=Math.atan2(ay,ax)+Math.PI/2;if(frameCount%3===0)addPart(t.x,t.y,'#888',0.5);}
  police.forEach(p=>{
    if(--p.pathTimer<=0){p.targetX=t.x;p.targetY=t.y;p.pathTimer=20;}
    const dx=p.targetX-p.x,dy=p.targetY-p.y,d=Math.sqrt(dx*dx+dy*dy)||1;
    const nx=p.x+dx/d*p.speed,ny=p.y+dy/d*p.speed;
    if(canMoveC(nx,p.y,12))p.x=nx;else p.pathTimer=0;
    if(canMoveC(p.x,ny,12))p.y=ny;else p.pathTimer=0;
    p.angle=Math.atan2(dy,dx)+Math.PI/2;
    if(frameCount%8===0)addPart(p.x,p.y-10,frameCount%16<8?'#f00':'#00f',0.6);
    const ex=p.x-t.x,ey=p.y-t.y;
    if(Math.sqrt(ex*ex+ey*ey)<26){lives--;for(let i=0;i<20;i++)addPart(t.x,t.y,'#ff6600',1);t.x=2*TILE+TILE/2;t.y=2*TILE+TILE/2;spawnPolice(Math.min(level+1,5));if(lives<=0)gameOver('\uD83D\uDEA8','Final Score: '+score+' \u2014 Level '+level);}
  });
  if(frameCount%30===0)score++;
  const nl=1+Math.floor(score/300);
  if(nl>level){level=nl;spawnPolice(Math.min(level+1,5));t.baseSpeed=2.8+level*0.15;}
  particles=particles.filter(p=>{p.x+=p.vx;p.y+=p.vy;p.life-=0.04;return p.life>0;});
}
function addPart(x,y,color,life){const a=Math.random()*Math.PI*2,s=Math.random()*2+0.5;particles.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,color,life,maxLife:life});}
function chaseDraw(){
  ctx.fillStyle='#1a1a1a';ctx.fillRect(0,0,CW,CH);
  for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){
    const t=MAP[r][c],x=c*TILE,y=r*TILE;
    if(t===0){
      ctx.fillStyle='#374151';ctx.fillRect(x,y,TILE,TILE);
      ctx.strokeStyle='#4b5563';ctx.lineWidth=1;ctx.setLineDash([6,6]);
      if(r>0&&r<ROWS-1&&MAP[r-1]?.[c]===0&&MAP[r+1]?.[c]===0){ctx.beginPath();ctx.moveTo(x+TILE/2,y);ctx.lineTo(x+TILE/2,y+TILE);ctx.stroke();}
      if(c>0&&c<COLS-1&&MAP[r]?.[c-1]===0&&MAP[r]?.[c+1]===0){ctx.beginPath();ctx.moveTo(x,y+TILE/2);ctx.lineTo(x+TILE,y+TILE/2);ctx.stroke();}
      ctx.setLineDash([]);
    }else if(t===1){
      ctx.fillStyle='#1e293b';ctx.fillRect(x,y,TILE,TILE);
      ctx.fillStyle='#0f172a';ctx.fillRect(x+3,y+3,TILE-6,TILE-6);
      const wc=['#fef08a','#fde047','#fbbf24','#f59e0b'];
      for(let wy=0;wy<3;wy++)for(let wx=0;wx<2;wx++){ctx.fillStyle=wc[(c*r+wx+wy)%wc.length];ctx.fillRect(x+8+wx*14,y+8+wy*10,7,6);}
    }else{ctx.fillStyle='#166534';ctx.fillRect(x,y,TILE,TILE);ctx.fillStyle='#16a34a';ctx.beginPath();ctx.arc(x+TILE/2,y+TILE/2,10,0,Math.PI*2);ctx.fill();}
  }
  ctx.fillStyle='rgba(255,255,255,0.08)';ctx.font='bold 70px sans-serif';ctx.textAlign='center';ctx.fillText('NYC',CW/2,CH/2+25);ctx.textAlign='left';
  particles.forEach(p=>{ctx.globalAlpha=p.life/p.maxLife;ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fill();});
  ctx.globalAlpha=1;
  drawCar(thief.x,thief.y,thief.angle,'#f5c518','#e63946');
  police.forEach(p=>drawCar(p.x,p.y,p.angle,'#1e3a8a','#93c5fd'));
}
function drawCar(x,y,angle,body,accent){
  ctx.save();ctx.translate(x,y);ctx.rotate(angle);
  ctx.fillStyle='rgba(0,0,0,0.25)';ctx.beginPath();ctx.ellipse(2,2,14,10,0,0,Math.PI*2);ctx.fill();
  ctx.fillStyle=body;ctx.beginPath();ctx.roundRect(-10,-16,20,32,4);ctx.fill();
  ctx.fillStyle=accent;ctx.beginPath();ctx.roundRect(-7,-10,14,16,3);ctx.fill();
  ctx.fillStyle='rgba(255,255,255,0.45)';ctx.beginPath();ctx.roundRect(-5,-9,10,6,2);ctx.fill();
  ctx.fillStyle='#fef9c3';ctx.fillRect(-9,-17,5,3);ctx.fillRect(4,-17,5,3);
  ctx.fillStyle='#ef4444';ctx.fillRect(-9,14,5,3);ctx.fillRect(4,14,5,3);
  ctx.restore();
}

// =============================================================
//  GAME 2 – SNAKE
// =============================================================
const SN=20;
let snake,snakeDir,snakeNext,food,snakeSpeed;
function startSnake(){
  getCanvas();canvas.width=600;canvas.height=500;
  score=0;lives=1;level=1;frameCount=0;snakeSpeed=8;
  snake=[{x:15,y:12},{x:14,y:12},{x:13,y:12}];
  snakeDir={x:1,y:0};snakeNext={x:1,y:0};
  placeFood();
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning=true;
  document.addEventListener('keydown',snakeKey);
  (function loop(){if(!gameRunning)return;raf=requestAnimationFrame(loop);frameCount++;if(frameCount%snakeSpeed===0)snakeStep();snakeDraw();updateHUD();})();
}
function snakeKey(e){
  const d={ArrowUp:{x:0,y:-1},ArrowDown:{x:0,y:1},ArrowLeft:{x:-1,y:0},ArrowRight:{x:1,y:0},w:{x:0,y:-1},s:{x:0,y:1},a:{x:-1,y:0},d:{x:1,y:0}};
  const nd=d[e.key];if(nd&&!(nd.x===-snakeDir.x&&nd.y===-snakeDir.y))snakeNext=nd;e.preventDefault();
}
function placeFood(){const cols=Math.floor(600/SN),rows=Math.floor(500/SN);do{food={x:Math.floor(Math.random()*cols),y:Math.floor(Math.random()*rows)};}while(snake.some(s=>s.x===food.x&&s.y===food.y));}
function snakeStep(){
  snakeDir={...snakeNext};
  const head={x:snake[0].x+snakeDir.x,y:snake[0].y+snakeDir.y};
  const cols=Math.floor(600/SN),rows=Math.floor(500/SN);
  if(head.x<0||head.y<0||head.x>=cols||head.y>=rows||snake.some(s=>s.x===head.x&&s.y===head.y))return gameOver('\uD83D\uDC0D','Score: '+score);
  snake.unshift(head);
  if(head.x===food.x&&head.y===food.y){score+=10;placeFood();snakeSpeed=Math.max(3,8-Math.floor(score/50));level=Math.floor(score/50)+1;}
  else snake.pop();
}
function snakeDraw(){
  ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,600,500);
  ctx.strokeStyle='#1a1a1a';ctx.lineWidth=0.5;
  for(let x=0;x<600;x+=SN){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,500);ctx.stroke();}
  for(let y=0;y<500;y+=SN){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(600,y);ctx.stroke();}
  ctx.shadowBlur=12;ctx.shadowColor='#ff4444';
  ctx.fillStyle='#ff4444';ctx.beginPath();ctx.arc(food.x*SN+SN/2,food.y*SN+SN/2,SN/2-2,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
  snake.forEach((s,i)=>{
    const t=i/snake.length;
    ctx.fillStyle='hsl('+(130-t*40)+',70%,'+(50-t*20)+'%)';
    ctx.beginPath();ctx.roundRect(s.x*SN+1,s.y*SN+1,SN-2,SN-2,i===0?6:3);ctx.fill();
    if(i===0){
      ctx.fillStyle='#fff';
      const ex=snakeDir.x,ey=snakeDir.y,cx=s.x*SN+SN/2,cy=s.y*SN+SN/2,ox=ey*4,oy=ex*4;
      ctx.beginPath();ctx.arc(cx+ex*3-ox,cy+ey*3-oy,2.5,0,Math.PI*2);ctx.fill();
      ctx.beginPath();ctx.arc(cx+ex*3+ox,cy+ey*3+oy,2.5,0,Math.PI*2);ctx.fill();
    }
  });
}

// =============================================================
//  GAME 3 – ASTEROIDS
// =============================================================
let ship,bullets,asteroids,astParticles;
function startAsteroids(){
  getCanvas();canvas.width=700;canvas.height=550;
  score=0;lives=3;level=1;frameCount=0;
  ship={x:350,y:275,angle:-Math.PI/2,vx:0,vy:0,invincible:120};
  bullets=[];asteroids=[];astParticles=[];
  spawnAsteroids(4);
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning=true;
  document.addEventListener('keydown',onKey);document.addEventListener('keyup',offKey);
  (function loop(){if(!gameRunning)return;raf=requestAnimationFrame(loop);frameCount++;astUpdate();astDraw();updateHUD();})();
}
function spawnAsteroids(n){
  for(let i=0;i<n;i++){
    const angle=Math.random()*Math.PI*2,edge=Math.random()<0.5;
    const x=edge?Math.random()*700:(Math.random()<0.5?0:700);
    const y=edge?(Math.random()<0.5?0:550):Math.random()*550;
    const spd=0.8+Math.random()*1.2+level*0.1;
    asteroids.push({x,y,vx:Math.cos(angle)*spd,vy:Math.sin(angle)*spd,r:30+Math.random()*20,angle:Math.random()*Math.PI*2,rot:0.01*(Math.random()-0.5),pts:genAstPts()});
  }
}
function genAstPts(){const n=8+Math.floor(Math.random()*5),pts=[];for(let i=0;i<n;i++){const a=i/n*Math.PI*2;pts.push({a,r:0.7+Math.random()*0.5});}return pts;}
function wrap(v,max){return((v%max)+max)%max;}
function astUpdate(){
  if(keys['ArrowLeft']||keys['a'])ship.angle-=0.06;
  if(keys['ArrowRight']||keys['d'])ship.angle+=0.06;
  if(keys['ArrowUp']||keys['w']){
    ship.vx+=Math.cos(ship.angle)*0.22;ship.vy+=Math.sin(ship.angle)*0.22;
    if(frameCount%2===0)addAstPart(ship.x-Math.cos(ship.angle)*14,ship.y-Math.sin(ship.angle)*14,'#ff8800',0.6,Math.cos(ship.angle+Math.PI)*(1+Math.random()),Math.sin(ship.angle+Math.PI)*(1+Math.random()));
  }
  ship.vx*=0.99;ship.vy*=0.99;
  ship.x=wrap(ship.x+ship.vx,700);ship.y=wrap(ship.y+ship.vy,550);
  if(ship.invincible>0)ship.invincible--;
  if(keys[' ']&&frameCount%12===0)bullets.push({x:ship.x+Math.cos(ship.angle)*16,y:ship.y+Math.sin(ship.angle)*16,vx:Math.cos(ship.angle)*7+ship.vx,vy:Math.sin(ship.angle)*7+ship.vy,life:60});
  bullets=bullets.filter(b=>{b.x=wrap(b.x+b.vx,700);b.y=wrap(b.y+b.vy,550);return--b.life>0;});
  asteroids.forEach(a=>{a.x=wrap(a.x+a.vx,700);a.y=wrap(a.y+a.vy,550);a.angle+=a.rot;});
  const toRemove=new Set();
  bullets.forEach((b,bi)=>{
    asteroids.forEach((a,ai)=>{
      if(Math.sqrt((b.x-a.x)**2+(b.y-a.y)**2)<a.r){
        score+=a.r>35?10:20;toRemove.add(bi);
        for(let i=0;i<8;i++)addAstPart(a.x,a.y,'#aaa',0.8);
        if(a.r>20){for(let i=0;i<2;i++){const ang=Math.random()*Math.PI*2,spd=1.5+Math.random();asteroids.push({x:a.x,y:a.y,vx:Math.cos(ang)*spd,vy:Math.sin(ang)*spd,r:a.r*0.55,angle:0,rot:a.rot*1.5,pts:genAstPts()});}}
        asteroids.splice(ai,1);
      }
    });
  });
  bullets=bullets.filter((_,i)=>!toRemove.has(i));
  if(asteroids.length===0){level++;spawnAsteroids(4+level);}
  if(ship.invincible===0){
    asteroids.forEach(a=>{
      if(Math.sqrt((ship.x-a.x)**2+(ship.y-a.y)**2)<a.r+12){
        lives--;for(let i=0;i<15;i++)addAstPart(ship.x,ship.y,'#ff6600',1);
        ship.x=350;ship.y=275;ship.vx=0;ship.vy=0;ship.invincible=120;
        if(lives<=0)gameOver('\uD83D\uDCA5','Score: '+score+' \u2014 Level '+level);
      }
    });
  }
  astParticles=astParticles.filter(p=>{p.x+=p.vx;p.y+=p.vy;p.life-=0.03;return p.life>0;});
}
function addAstPart(x,y,color,life,vx,vy){const a=Math.random()*Math.PI*2,s=Math.random()*2+0.5;astParticles.push({x,y,vx:vx??Math.cos(a)*s,vy:vy??Math.sin(a)*s,color,life,maxLife:life});}
function astDraw(){
  ctx.fillStyle='#040408';ctx.fillRect(0,0,700,550);
  ctx.fillStyle='rgba(255,255,255,0.5)';
  for(let i=0;i<80;i++)ctx.fillRect((i*137+42)%700,(i*97+17)%550,1,1);
  asteroids.forEach(a=>{
    ctx.save();ctx.translate(a.x,a.y);ctx.rotate(a.angle);
    ctx.strokeStyle='#aaa';ctx.lineWidth=2;ctx.fillStyle='#2a2a3a';
    ctx.beginPath();a.pts.forEach((p,i)=>{const x=Math.cos(p.a)*a.r*p.r,y=Math.sin(p.a)*a.r*p.r;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();
  });
  ctx.fillStyle='#ffff00';ctx.shadowBlur=8;ctx.shadowColor='#ffff00';
  bullets.forEach(b=>{ctx.beginPath();ctx.arc(b.x,b.y,3,0,Math.PI*2);ctx.fill();});
  ctx.shadowBlur=0;
  astParticles.forEach(p=>{ctx.globalAlpha=p.life/p.maxLife;ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;
  if(ship.invincible>0&&frameCount%6<3)return;
  ctx.save();ctx.translate(ship.x,ship.y);ctx.rotate(ship.angle+Math.PI/2);
  ctx.strokeStyle='#00ff88';ctx.lineWidth=2;ctx.fillStyle='rgba(0,255,136,0.15)';ctx.shadowBlur=10;ctx.shadowColor='#00ff88';
  ctx.beginPath();ctx.moveTo(0,-18);ctx.lineTo(-10,12);ctx.lineTo(0,6);ctx.lineTo(10,12);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();
}

// =============================================================
//  GAME 4 – BREAKOUT
// =============================================================
let paddle,ball,bricks;
const BW=700,BH=500,BRICK_ROWS=5,BRICK_COLS=12;
function startBreakout(){
  getCanvas();canvas.width=BW;canvas.height=BH;
  score=0;lives=3;level=1;frameCount=0;
  paddle={x:BW/2,y:BH-30,w:90,h:10};
  ball={x:BW/2,y:BH-50,vx:3.5,vy:-3.5,r:8};
  buildBricks();
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning=true;
  document.addEventListener('keydown',onKey);document.addEventListener('keyup',offKey);
  canvas.addEventListener('mousemove',brkMouse);
  (function loop(){if(!gameRunning)return;raf=requestAnimationFrame(loop);frameCount++;brkUpdate();brkDraw();updateHUD();})();
}
function buildBricks(){
  bricks=[];const colors=['#e63946','#f4a261','#e9c46a','#2a9d8f','#457b9d'];
  for(let r=0;r<BRICK_ROWS;r++)for(let c=0;c<BRICK_COLS;c++)bricks.push({x:c*58+4,y:r*24+40,w:56,h:22,alive:true,color:colors[r%colors.length]});
}
function brkMouse(e){const rect=canvas.getBoundingClientRect();paddle.x=Math.max(paddle.w/2,Math.min(BW-paddle.w/2,e.clientX-rect.left));}
function brkUpdate(){
  if(keys['ArrowLeft'])paddle.x=Math.max(paddle.w/2,paddle.x-6);
  if(keys['ArrowRight'])paddle.x=Math.min(BW-paddle.w/2,paddle.x+6);
  ball.x+=ball.vx;ball.y+=ball.vy;
  if(ball.x-ball.r<0||ball.x+ball.r>BW)ball.vx*=-1;
  if(ball.y-ball.r<0)ball.vy*=-1;
  if(ball.y+ball.r>BH){lives--;ball.x=BW/2;ball.y=BH-50;ball.vx=3.5;ball.vy=-3.5;if(lives<=0)gameOver('\uD83E\uDDF1','Score: '+score);}
  if(ball.y+ball.r>paddle.y&&ball.y+ball.r<paddle.y+paddle.h+4&&ball.x>paddle.x-paddle.w/2&&ball.x<paddle.x+paddle.w/2){ball.vy=-Math.abs(ball.vy);ball.vx=((ball.x-paddle.x)/(paddle.w/2))*5;}
  let allDead=true;
  bricks.forEach(b=>{
    if(!b.alive)return;allDead=false;
    if(ball.x+ball.r>b.x&&ball.x-ball.r<b.x+b.w&&ball.y+ball.r>b.y&&ball.y-ball.r<b.y+b.h){
      b.alive=false;score+=10;
      if(Math.abs(ball.x-(b.x+b.w/2))/b.w>Math.abs(ball.y-(b.y+b.h/2))/b.h)ball.vx*=-1;else ball.vy*=-1;
    }
  });
  if(allDead){level++;buildBricks();ball.vy=-(Math.abs(ball.vy)+0.3);ball.vx+=ball.vx>0?0.2:-0.2;}
}
function brkDraw(){
  ctx.fillStyle='#06060f';ctx.fillRect(0,0,BW,BH);
  bricks.forEach(b=>{if(!b.alive)return;ctx.fillStyle=b.color;ctx.beginPath();ctx.roundRect(b.x,b.y,b.w,b.h,4);ctx.fill();ctx.fillStyle='rgba(255,255,255,0.2)';ctx.fillRect(b.x+2,b.y+2,b.w-4,5);});
  const pg=ctx.createLinearGradient(paddle.x-paddle.w/2,0,paddle.x+paddle.w/2,0);pg.addColorStop(0,'#4488ff');pg.addColorStop(1,'#88ccff');
  ctx.fillStyle=pg;ctx.beginPath();ctx.roundRect(paddle.x-paddle.w/2,paddle.y,paddle.w,paddle.h,5);ctx.fill();
  ctx.shadowBlur=14;ctx.shadowColor='#fff';ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(ball.x,ball.y,ball.r,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
}

// =============================================================
//  GAME 5 – FLAPPY BIRD
// =============================================================
const FW=400,FH=550,GAP=130,PIPE_W=52,PIPE_SPD=2.2;
let bird,pipes,pipeTimer,bgOffset;
function startFlappy(){
  getCanvas();canvas.width=FW;canvas.height=FH;
  score=0;lives=1;level=1;frameCount=0;bgOffset=0;
  bird={x:80,y:FH/2,vy:0,angle:0};
  pipes=[];pipeTimer=60;
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning=true;
  document.addEventListener('keydown',flappyKey);
  canvas.addEventListener('click',flap);
  (function loop(){if(!gameRunning)return;raf=requestAnimationFrame(loop);frameCount++;flappyUpdate();flappyDraw();updateHUD();})();
}
function flappyKey(e){if(e.key===' '||e.key==='ArrowUp'||e.key==='w'){flap();e.preventDefault();}}
function flap(){bird.vy=-7;}
function flappyUpdate(){
  bird.vy+=0.38;bird.y+=bird.vy;bird.angle=Math.max(-0.5,Math.min(1.2,bird.vy*0.08));
  bgOffset=(bgOffset+0.5)%FW;
  if(++pipeTimer>90){pipeTimer=0;const top=60+Math.random()*(FH-GAP-120);pipes.push({x:FW,topH:top,scored:false});}
  pipes=pipes.filter(p=>{
    p.x-=PIPE_SPD+(level-1)*0.15;
    if(!p.scored&&p.x+PIPE_W<bird.x){p.scored=true;score++;if(score%10===0)level++;}
    const inX=bird.x+10>p.x&&bird.x-10<p.x+PIPE_W;
    if(inX&&(bird.y-10<p.topH||bird.y+10>p.topH+GAP))gameOver('\uD83D\uDC26','Score: '+score);
    return p.x>-PIPE_W;
  });
  if(bird.y<0||bird.y>FH)gameOver('\uD83D\uDC26','Score: '+score);
}
function flappyDraw(){
  const sg=ctx.createLinearGradient(0,0,0,FH);sg.addColorStop(0,'#87ceeb');sg.addColorStop(1,'#c9e8f5');ctx.fillStyle=sg;ctx.fillRect(0,0,FW,FH);
  ctx.fillStyle='rgba(255,255,255,0.8)';
  [[60,80],[160,50],[280,100],[380,65]].forEach(([cx,cy])=>{const ox=((cx-bgOffset*0.3)%FW+FW)%FW;ctx.beginPath();ctx.arc(ox,cy,22,0,Math.PI*2);ctx.arc(ox+20,cy-6,16,0,Math.PI*2);ctx.arc(ox+38,cy,18,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle='#5d8a3c';ctx.fillRect(0,FH-40,FW,40);ctx.fillStyle='#3d6b2a';ctx.fillRect(0,FH-40,FW,8);
  pipes.forEach(p=>{
    const g=ctx.createLinearGradient(p.x,0,p.x+PIPE_W,0);g.addColorStop(0,'#2d6a2d');g.addColorStop(0.4,'#4caf50');g.addColorStop(1,'#1a4a1a');ctx.fillStyle=g;
    ctx.fillRect(p.x,0,PIPE_W,p.topH);ctx.fillRect(p.x-4,p.topH-20,PIPE_W+8,20);
    const botY=p.topH+GAP;ctx.fillRect(p.x,botY,PIPE_W,FH-botY);ctx.fillRect(p.x-4,botY,PIPE_W+8,20);
  });
  ctx.save();ctx.translate(bird.x,bird.y);ctx.rotate(bird.angle);
  ctx.fillStyle='#f5c518';ctx.beginPath();ctx.ellipse(0,0,15,11,0,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#e6b800';ctx.beginPath();ctx.ellipse(-4,frameCount%12<6?4:8,8,5,0.4,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(8,-3,5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#222';ctx.beginPath();ctx.arc(9,-3,2.5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#ff8800';ctx.beginPath();ctx.moveTo(14,1);ctx.lineTo(22,3);ctx.lineTo(14,6);ctx.closePath();ctx.fill();
  ctx.restore();
}

// Exports
window.launchGame        = launchGame;
window.stopGame          = stopGame;
window.startCurrentGame  = startCurrentGame;
window.showLauncher      = showLauncher;

// =============================================================
//  GAME 6 – CAT RUN!
//  Top-down neighbourhood. You are a cat; avoid the catchers.
//  Collect fish for points. WASD / Arrows to move.
// =============================================================
const CR_W = 640, CR_H = 560, CR_TILE = 40;

// 0=grass, 1=pavement/road, 2=house, 3=fence, 4=bush
const CR_MAP = [
  [2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2],
  [2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [2,2,1,0,0,0,1,0,4,0,1,0,0,0,1,2],
  [2,2,1,0,4,0,1,0,0,0,1,4,0,0,1,2],
  [2,2,1,0,0,0,1,0,0,4,1,0,0,0,1,2],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [0,0,1,0,0,4,1,2,2,2,1,4,0,0,0,1],
  [4,0,1,0,0,0,1,2,2,2,1,0,0,4,0,1],
  [0,0,1,4,0,0,1,2,2,2,1,0,0,0,4,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [2,2,1,0,4,0,1,0,0,0,1,0,4,0,1,2],
  [2,2,1,0,0,0,1,4,0,0,1,0,0,0,1,2],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
];

function crWalkable(c, r) {
  if (c < 0 || r < 0 || c >= CR_W/CR_TILE || r >= CR_H/CR_TILE) return false;
  const t = CR_MAP[r]?.[c];
  return t === 0 || t === 1; // grass or road
}
function crCanMove(x, y, half) {
  return [
    [x-half, y-half],[x+half, y-half],
    [x-half, y+half],[x+half, y+half],
  ].every(([cx,cy]) => crWalkable(Math.floor(cx/CR_TILE), Math.floor(cy/CR_TILE)));
}

let cat, catchers, fish, crParticles, crSpeedMult;

function startCatRun() {
  getCanvas(); canvas.width = CR_W; canvas.height = CR_H;
  score = 0; lives = 3; level = 1; frameCount = 0;
  crSpeedMult = 1;
  cat = { x: 2*CR_TILE+CR_TILE/2, y: 2*CR_TILE+CR_TILE/2, angle: 0, spd: 3.0 };
  crParticles = [];
  spawnCatchers(1);
  spawnFish(4);
  document.getElementById('game-msg').classList.add('hidden');
  gameRunning = true;
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup',   offKey);
  (function loop() {
    if (!gameRunning) return;
    raf = requestAnimationFrame(loop);
    frameCount++;
    crUpdate();
    crDraw();
    updateHUD();
  })();
}

function spawnCatchers(n) {
  catchers = [];
  const starts = [
    {x:13*CR_TILE+CR_TILE/2, y:2*CR_TILE+CR_TILE/2},
    {x:13*CR_TILE+CR_TILE/2, y:11*CR_TILE+CR_TILE/2},
    {x:2*CR_TILE+CR_TILE/2,  y:11*CR_TILE+CR_TILE/2},
    {x:7*CR_TILE+CR_TILE/2,  y:6*CR_TILE+CR_TILE/2},
  ];
  for (let i = 0; i < n; i++) {
    const s = starts[i % starts.length];
    catchers.push({ x: s.x, y: s.y, angle: 0, spd: 1.4 + level * 0.2, pathTimer: 0 });
  }
}

function spawnFish(n) {
  fish = [];
  const grassCells = [];
  for (let r = 0; r < CR_H/CR_TILE; r++)
    for (let c = 0; c < CR_W/CR_TILE; c++)
      if (CR_MAP[r]?.[c] === 0) grassCells.push({c,r});
  for (let i = 0; i < n; i++) {
    const cell = grassCells[Math.floor(Math.random() * grassCells.length)];
    fish.push({ x: cell.c*CR_TILE+CR_TILE/2, y: cell.r*CR_TILE+CR_TILE/2, eaten: false });
  }
}

function crUpdate() {
  const spd = cat.spd;
  let ax = 0, ay = 0;
  if (keys['ArrowLeft']  || keys['a'] || keys['A']) ax = -spd;
  if (keys['ArrowRight'] || keys['d'] || keys['D']) ax =  spd;
  if (keys['ArrowUp']    || keys['w'] || keys['W']) ay = -spd;
  if (keys['ArrowDown']  || keys['s'] || keys['S']) ay =  spd;
  if (ax && ay) { ax *= 0.707; ay *= 0.707; }
  if (crCanMove(cat.x + ax, cat.y, 14)) cat.x += ax;
  if (crCanMove(cat.x, cat.y + ay, 14)) cat.y += ay;
  if (ax || ay) cat.angle = Math.atan2(ay, ax);

  // Catchers chase
  catchers.forEach(ch => {
    if (--ch.pathTimer <= 0) { ch.pathTimer = 15; }
    const dx = cat.x - ch.x, dy = cat.y - ch.y;
    const d = Math.sqrt(dx*dx + dy*dy) || 1;
    const nx = ch.x + (dx/d)*ch.spd, ny = ch.y + (dy/d)*ch.spd;
    if (crCanMove(nx, ch.y, 12)) ch.x = nx; else ch.pathTimer = 0;
    if (crCanMove(ch.x, ny, 12)) ch.y = ny; else ch.pathTimer = 0;
    ch.angle = Math.atan2(dy, dx);

    // Net swing particle
    if (frameCount % 10 === 0) crAddPart(ch.x + Math.cos(ch.angle)*18, ch.y + Math.sin(ch.angle)*18, '#88ddff', 0.5);

    // Catch!
    if (Math.sqrt((ch.x-cat.x)**2 + (ch.y-cat.y)**2) < 24) {
      lives--;
      for (let i = 0; i < 16; i++) crAddPart(cat.x, cat.y, '#ff9900', 1);
      cat.x = 2*CR_TILE+CR_TILE/2; cat.y = 2*CR_TILE+CR_TILE/2;
      spawnCatchers(Math.min(level + 1, 4));
      if (lives <= 0) gameOver('\uD83D\uDC31', 'Caught! Score: ' + score);
    }
  });

  // Eat fish
  fish.forEach(f => {
    if (f.eaten) return;
    if (Math.sqrt((f.x-cat.x)**2 + (f.y-cat.y)**2) < 20) {
      f.eaten = true; score += 15;
      for (let i = 0; i < 10; i++) crAddPart(f.x, f.y, '#ffe066', 0.8);
    }
  });
  // Respawn fish when all eaten
  if (fish.every(f => f.eaten)) { level++; spawnFish(4 + level); spawnCatchers(Math.min(level + 1, 4)); cat.spd = 3.0 + level * 0.1; }

  // Score tick
  if (frameCount % 60 === 0) score++;

  crParticles = crParticles.filter(p => { p.x+=p.vx; p.y+=p.vy; p.life-=0.05; return p.life>0; });
}

function crAddPart(x, y, color, life) {
  const a = Math.random()*Math.PI*2, s = Math.random()*2+0.5;
  crParticles.push({ x, y, vx: Math.cos(a)*s, vy: Math.sin(a)*s, color, life, maxLife: life });
}

function crDraw() {
  // Tiles
  for (let r = 0; r < CR_H/CR_TILE; r++) {
    for (let c = 0; c < CR_W/CR_TILE; c++) {
      const t = CR_MAP[r]?.[c] ?? 0;
      const x = c*CR_TILE, y = r*CR_TILE;
      if (t === 0) {
        ctx.fillStyle = '#4a7c3f'; ctx.fillRect(x, y, CR_TILE, CR_TILE);
        // grass texture dots
        ctx.fillStyle = '#3d6b33';
        ctx.fillRect(x+6, y+6, 3, 3); ctx.fillRect(x+20, y+22, 3, 3); ctx.fillRect(x+30, y+10, 3, 3);
      } else if (t === 1) {
        ctx.fillStyle = '#a0a0a0'; ctx.fillRect(x, y, CR_TILE, CR_TILE);
        ctx.strokeStyle = '#888'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(x+CR_TILE/2, y); ctx.lineTo(x+CR_TILE/2, y+CR_TILE); ctx.stroke();
        ctx.setLineDash([]);
      } else if (t === 2) {
        // House
        ctx.fillStyle = '#c0392b'; ctx.fillRect(x, y, CR_TILE, CR_TILE);
        ctx.fillStyle = '#922b21'; ctx.fillRect(x+3, y+3, CR_TILE-6, CR_TILE-6);
        ctx.fillStyle = '#ffeaa7'; ctx.fillRect(x+8, y+8, 10, 10); ctx.fillRect(x+22, y+8, 10, 10);
        ctx.fillStyle = '#6c5ce7'; ctx.fillRect(x+13, y+20, 14, CR_TILE-20);
      } else if (t === 3) {
        ctx.fillStyle = '#8B6914'; ctx.fillRect(x, y, CR_TILE, CR_TILE);
        ctx.fillStyle = '#5D4037'; ctx.fillRect(x+15, y, 10, CR_TILE);
      } else if (t === 4) {
        ctx.fillStyle = '#4a7c3f'; ctx.fillRect(x, y, CR_TILE, CR_TILE);
        ctx.fillStyle = '#27ae60'; ctx.beginPath(); ctx.arc(x+CR_TILE/2, y+CR_TILE/2, 14, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#1e8449'; ctx.beginPath(); ctx.arc(x+CR_TILE/2-5, y+CR_TILE/2+5, 10, 0, Math.PI*2); ctx.fill();
      }
    }
  }

  // Fish pickups
  fish.forEach(f => {
    if (f.eaten) return;
    const bob = Math.sin(frameCount * 0.08 + f.x) * 3;
    ctx.save(); ctx.translate(f.x, f.y + bob);
    // Body
    ctx.fillStyle = '#74b9ff'; ctx.beginPath(); ctx.ellipse(0, 0, 10, 6, 0, 0, Math.PI*2); ctx.fill();
    // Tail
    ctx.fillStyle = '#0984e3';
    ctx.beginPath(); ctx.moveTo(-10,0); ctx.lineTo(-16,-6); ctx.lineTo(-16,6); ctx.closePath(); ctx.fill();
    // Eye
    ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(6,-1,2.5,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#222'; ctx.beginPath(); ctx.arc(6.5,-1,1.2,0,Math.PI*2); ctx.fill();
    // Glow
    ctx.shadowBlur = 10; ctx.shadowColor = '#74b9ff';
    ctx.strokeStyle = 'rgba(116,185,255,0.5)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.ellipse(0, 0, 13, 9, 0, 0, Math.PI*2); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.restore();
  });

  // Particles
  crParticles.forEach(p => {
    ctx.globalAlpha = p.life / p.maxLife;
    ctx.fillStyle = p.color;
    ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI*2); ctx.fill();
  }); ctx.globalAlpha = 1;

  // Catchers (person with net)
  catchers.forEach(ch => drawCatcher(ch.x, ch.y, ch.angle));

  // Cat (player)
  drawCat(cat.x, cat.y, cat.angle, ax => ax, ay => ay);
}

function drawCat(x, y, angle) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle + Math.PI/2);
  const run = Math.floor(frameCount / 5) % 4;

  // Tail (behind body) – wavy
  ctx.strokeStyle = '#e17055'; ctx.lineWidth = 4; ctx.lineCap = 'round';
  ctx.beginPath();
  const tailWave = Math.sin(frameCount * 0.15) * 8;
  ctx.moveTo(0, 12);
  ctx.bezierCurveTo(-8, 20+tailWave, 10+tailWave, 32, 5+tailWave, 38);
  ctx.stroke();

  // Body
  ctx.fillStyle = '#e17055';
  ctx.beginPath(); ctx.ellipse(0, 4, 11, 14, 0, 0, Math.PI*2); ctx.fill();

  // Stripes
  ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 2;
  [-4,0,4].forEach(sy => { ctx.beginPath(); ctx.moveTo(-9, sy); ctx.lineTo(9, sy); ctx.stroke(); });

  // Head
  ctx.fillStyle = '#e17055';
  ctx.beginPath(); ctx.arc(0, -12, 11, 0, Math.PI*2); ctx.fill();

  // Ears
  ctx.fillStyle = '#e17055';
  ctx.beginPath(); ctx.moveTo(-9,-18); ctx.lineTo(-14,-26); ctx.lineTo(-3,-20); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(9,-18); ctx.lineTo(14,-26); ctx.lineTo(3,-20); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#fab1a0';
  ctx.beginPath(); ctx.moveTo(-8,-19); ctx.lineTo(-12,-25); ctx.lineTo(-4,-20); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(8,-19); ctx.lineTo(12,-25); ctx.lineTo(4,-20); ctx.closePath(); ctx.fill();

  // Eyes (blinking)
  const blink = frameCount % 90 < 6;
  ctx.fillStyle = '#fdcb6e';
  if (!blink) { ctx.beginPath(); ctx.ellipse(-4,-13,4,4.5,0,0,Math.PI*2); ctx.fill(); ctx.beginPath(); ctx.ellipse(4,-13,4,4.5,0,0,Math.PI*2); ctx.fill(); }
  else        { ctx.fillRect(-7,-13,6,2); ctx.fillRect(2,-13,6,2); }
  ctx.fillStyle = '#2d3436';
  if (!blink) { ctx.beginPath(); ctx.ellipse(-4,-13,2,3,0,0,Math.PI*2); ctx.fill(); ctx.beginPath(); ctx.ellipse(4,-13,2,3,0,0,Math.PI*2); ctx.fill(); }
  // Eye shine
  ctx.fillStyle = '#fff';
  if (!blink) { ctx.beginPath(); ctx.arc(-3,-14,1,0,Math.PI*2); ctx.fill(); ctx.beginPath(); ctx.arc(5,-14,1,0,Math.PI*2); ctx.fill(); }

  // Nose
  ctx.fillStyle = '#fd79a8'; ctx.beginPath(); ctx.arc(0,-10,2.5,0,Math.PI*2); ctx.fill();
  // Whiskers
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1;
  [[-12,-10,-4,-9],[12,-10,4,-9],[-12,-8,-4,-9],[12,-8,4,-9]].forEach(([x1,y1,x2,y2]) => { ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); });

  // Legs (running animation)
  const legOff = [0,1,2,3].map(i => Math.sin((frameCount*0.25)+i*Math.PI/2)*5);
  ctx.fillStyle = '#c0392b';
  ctx.beginPath(); ctx.ellipse(-6, 14+legOff[0], 4, 6, 0.2, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(6,  14+legOff[1], 4, 6,-0.2, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(-5, 20+legOff[2], 3, 5, 0.2, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(5,  20+legOff[3], 3, 5,-0.2, 0, Math.PI*2); ctx.fill();

  ctx.restore();
}

function drawCatcher(x, y, angle) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle + Math.PI/2);

  // Body
  ctx.fillStyle = '#2980b9';
  ctx.beginPath(); ctx.roundRect(-8, -10, 16, 22, 4); ctx.fill();
  // Head
  ctx.fillStyle = '#ffeaa7';
  ctx.beginPath(); ctx.arc(0, -18, 10, 0, Math.PI*2); ctx.fill();
  // Hat
  ctx.fillStyle = '#1a252f';
  ctx.fillRect(-11, -26, 22, 5);
  ctx.fillRect(-7, -36, 14, 12);
  // Eyes
  ctx.fillStyle = '#2d3436';
  ctx.beginPath(); ctx.arc(-3,-18,2,0,Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(3,-18,2,0,Math.PI*2); ctx.fill();
  // Legs
  ctx.fillStyle = '#1a252f';
  const lw = Math.sin(frameCount*0.2)*6;
  ctx.fillRect(-7, 12, 6, 10+lw);
  ctx.fillRect(1, 12, 6, 10-lw);

  // Net pole
  ctx.strokeStyle = '#8B6914'; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(8, -8); ctx.lineTo(22, -22); ctx.stroke();
  // Net hoop
  ctx.strokeStyle = '#88ddff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(22, -22, 10, 0, Math.PI*2); ctx.stroke();
  // Net mesh
  ctx.globalAlpha = 0.4;
  for (let i = 0; i < 4; i++) {
    const na = i / 4 * Math.PI;
    ctx.beginPath(); ctx.moveTo(22, -22); ctx.lineTo(22+Math.cos(na)*10, -22+Math.sin(na)*10); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.restore();
}
