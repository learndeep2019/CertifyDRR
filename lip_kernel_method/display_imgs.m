% display figures
function display_imgs(imgs, name)
  if ismatrix(imgs)
    [features, num] = size(imgs);
    columns = sqrt(features);
    imgs = reshape(imgs, columns, columns, []);
  end
  
  [columns, rows, num] = size(imgs);
  rgbimgs = repmat(imgs, 1,3,1);
  rgbimgs = reshape(rgbimgs, columns, rows, 3, num);
  figure; 
  montage(rgbimgs);
  title(name, 'fontsize',16);
%   saveas(gcf, ['imgs_' name '.pdf']);
end

