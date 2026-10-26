"""Extract a genuine final recorded frame and summarize all local attempts."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args()
r=json.loads((a.episode/'report.json').read_text())
reader=imageio.get_reader(a.episode/'main_wrist.mp4')
frame=reader.get_data(reader.count_frames()-1);reader.close()
imageio.imwrite(a.episode/'final_frame.png',frame)
print(json.dumps({k:r.get(k) for k in ('success','error','steps','hang_part','cycles','code_changed_during_episode')},indent=2))
