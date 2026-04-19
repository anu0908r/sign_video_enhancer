import imageio

reader = imageio.get_reader('output1/output_vidstab.mp4')
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer('output1/output_vidstab_h264.mp4', fps=fps, codec='libx264')

for im in reader:
    writer.append_data(im)

writer.close()
print("Converted to H264 successfully.")
